from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.agent.tools import AgentTools
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from typing import TypedDict, List
from langchain_community.chat_models.tongyi import ChatTongyi
from typing import Any, Dict, List, Optional, Union
from langchain_core.documents import Document
import re, json
from langchain_core.language_models.chat_models import BaseChatModel
from MedicalRag.prompts.templates import get_prompt_template
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableMap
)
from functools import partial
from threading import Thread
from contextlib import nullcontext
from .tools import tavily_search
import logging
from ..core.utils import create_llm_client
from ..config.models import AppConfig
from ..retrieval.service import OnlineRetrievalService

logger = logging.getLogger(__name__)

WEB_ROUTER_TIMEOUT_SECONDS = 60


def _emit_local_trace(state: dict, event: str, **data) -> None:
    """只有 Eval Runner 注入收集器时才记录，正常 API Runtime 不持久化轨迹。"""
    collector = state.get("local_trace_collector")
    if collector is not None:
        collector.emit(state.get("local_trace_context"), event, **data)


def invoke_with_timeout(callable_obj, timeout_seconds: float):
    """为不受 SDK timeout 完整约束的同步模型调用提供调用层硬上限。"""
    result_box = []
    error_box = []

    def run() -> None:
        try:
            result_box.append(callable_obj())
        except Exception as error:
            error_box.append(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError(f"调用超过 {timeout_seconds} 秒")
    if error_box:
        raise error_box[0]
    return result_box[0]

def del_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def parse_web_search_response(text: str) -> tuple[list[Document], dict | None]:
    """解析 Web 工具的受控 JSON 返回值，并兼容旧版仅返回文档数组的格式。"""
    if not text or not text.strip():
        return [], {"code": "empty_response", "message": "联网搜索未返回内容。"}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("联网搜索工具返回了无效 JSON，已保留本地检索结果。")
        return [], {"code": "invalid_response", "message": "联网搜索服务返回格式无效。"}

    # 兼容项目改造前 web_search 直接返回的 Document 数组。
    if isinstance(payload, list):
        documents = payload
    elif isinstance(payload, dict):
        if not payload.get("ok"):
            error = payload.get("error") or {}
            return [], {
                "code": str(error.get("code", "provider_error")),
                "message": str(error.get("message", "联网搜索服务暂时不可用。")),
            }
        documents = payload.get("documents", [])
    else:
        return [], {"code": "invalid_response", "message": "联网搜索服务返回格式无效。"}

    if not isinstance(documents, list):
        return [], {"code": "invalid_response", "message": "联网搜索文档格式无效。"}
    try:
        return [Document(**document) for document in documents], None
    except (TypeError, ValueError):
        logger.warning("联网搜索工具返回的文档格式无效，已保留本地检索结果。")
        return [], {"code": "invalid_document", "message": "联网搜索文档格式无效。"}


def json_to_list_document(text: str) -> list:
    """保留旧函数名，供既有调用方只获取解析后的文档。"""
    documents, _ = parse_web_search_response(text)
    return documents


def merge_web_tool_responses(tool_messages: list[ToolMessage]) -> tuple[list[Document], list[dict], int]:
    """消费同一轮全部 web_search ToolMessage，保留所有成功结果并去重。"""
    documents: list[Document] = []
    errors: list[dict] = []
    success_count = 0
    seen: set[tuple[str, str]] = set()
    for message in tool_messages:
        parsed_documents, error = parse_web_search_response(str(message.content))
        if error is not None:
            errors.append(error)
            continue
        success_count += 1
        for document in parsed_documents:
            identity = (str(document.metadata.get("url", "")), document.page_content)
            if identity not in seen:
                seen.add(identity)
                documents.append(document)
    return documents, errors, success_count

def format_document_str(documents: List[Document]) -> str:
    parts = []
    for i, d in enumerate(reversed(documents)):
        if i >= 6:  # 做一个简单的截断
            break
        parts.append(f"## 文档{i+1}：\n{d.page_content}\n")
    return "".join(parts)
    

# ===================== 状态结构 =====================
class SearchMessagesState(TypedDict, total=False):
    query: str
    main_messages: List[Union[HumanMessage, AIMessage]]
    other_messages: List[BaseMessage]
    docs: List[Document]
    summary: str
    retry: int              # 剩余可重试次数
    final: str              # 最终输出
    judge_result: str
    retrieval_info: dict    # Stage 4：Stage 2 检索主链的可审计信息
    web_search_count: int
    web_search_status: dict
    judge_retry_count: int
    node_events: List[str]
    # 仅由 Stage 4 Eval Runner 注入；普通 API Runtime 不提供也不落盘。
    local_trace_collector: Any
    local_trace_context: dict
    
    
class NetworkSearchResult(BaseModel):
    need_search: bool = Field(description="是否需要进行网络搜索")
    search_query: str = Field(description="网络搜索查询词", default="")
    remain_doc_index: List[int] = Field(description="保留的文档索引列表", default=[])


def _should_call_tool(last_ai: BaseMessage) -> bool:
    """ 判断上一步是否触发了工具 """
    return bool(getattr(last_ai, "tool_calls", None))

def retrieve(
    state: SearchMessagesState,
    retrieval_service: OnlineRetrievalService,
    show_debug: bool,
) -> SearchMessagesState:
    """唯一在线本地检索节点：Planner → Executor → Reranker → Final Top-5。"""
    collector = state.get("local_trace_collector")
    context = state.get("local_trace_context")
    if collector is None:
        result = retrieval_service.retrieve(state["query"])
    else:
        result = retrieval_service.retrieve(state["query"], trace_collector=collector, trace_context=context)
    state["docs"] = list(result.documents)
    state["retrieval_info"] = {
        "selected_channels": list(result.selected_channels),
        "planner_error": result.planner_error,
        "candidate_count": result.candidate_count,
        "planner_latency_ms": result.planner_latency_ms,
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "reranker_latency_ms": result.reranker_latency_ms,
        "db_search_count": 1,
    }
    state.setdefault("node_events", []).extend(["retrieval_planner", "retrieval_executor", "reranker"])
    _emit_local_trace(state, "retrieval_planner", selected_channels=list(result.selected_channels),
                      planner_error=result.planner_error, latency_ms=result.planner_latency_ms)
    _emit_local_trace(state, "retrieval_executor", candidate_count=result.candidate_count,
                      candidate_pks=list(getattr(result, "candidate_pks", ())),
                      latency_ms=result.retrieval_latency_ms)
    _emit_local_trace(state, "reranker", final_count=len(result.documents),
                      final_pks=list(getattr(result, "final_pks", ())), latency_ms=result.reranker_latency_ms)
    if show_debug:
        logger.info("Stage 2 检索完成：通道=%s，候选=%s，最终=%s", result.selected_channels,
                    result.candidate_count, len(result.documents))
    return state


def llm_network_search(
    state: SearchMessagesState,
    judge_llm: BaseChatModel,
    network_search_llm: BaseChatModel,
    network_tool_node: ToolNode,
    show_debug: bool
) -> SearchMessagesState:
    """ 联网检索节点  可能触发web_tool """
    if show_debug:
        logger.info(f"检查是否缺失资料需要网络搜索...")
    # 创建Pydantic解析器
    parser = PydanticOutputParser(pydantic_object=NetworkSearchResult)
    # 可选：创建容错解析器，能自动修复格式错误
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=judge_llm)  # 使用不绑定工具的LLM
    
    # 获取格式指令并转义大括号
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    
    # 构建判断消息模板
    judge_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("web_router")["system"].format(format_instructions=format_instructions)), 
        ("human", get_prompt_template("web_router")["user"])
    ])
    
    # 构建工具调用消息模板
    calling_messages = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("call_web")["system"]),
        ("human", get_prompt_template("call_web")["user"])
    ])
    
    # 步骤1：判断是否需要搜索
    judge_chain = judge_messages | judge_llm
    
    try:
        # 执行判断链，直接得到解析后的结果
        collector = state.get("local_trace_collector")
        context = state.get("local_trace_context")
        span = (lambda name, **data: collector.span(context, name, **data)) if collector else None
        with (span("web_judge", provider="deepseek", operation="web_need_judge") if span else nullcontext()):
            raw_judge = invoke_with_timeout(
                lambda: judge_chain.invoke({"query": state['query'], "docs": format_document_str(state.get('docs', []))}),
                WEB_ROUTER_TIMEOUT_SECONDS,
            )
        with (span("output_parse_or_fix", provider="deepseek", operation="structured_output_fix") if span else nullcontext()):
            result: NetworkSearchResult = fixing_parser.parse(del_think(raw_judge.content))
        if show_debug:
            logger.info(f"判断结果: {'需要网络检索' if result.need_search else '不需要网络检索'}, 检索文本：{result.search_query}")
        
        # 保存判断消息到状态（用于调试）
        judge_ai_content = f"分析结果: {result.model_dump()}"
        judge_ai = AIMessage(content=judge_ai_content)
        state["other_messages"].append(judge_ai)
        state.setdefault("node_events", []).append("web_router")
        state["web_router"] = {
            "need_search": result.need_search,
            "search_query": result.search_query,
        }
        _emit_local_trace(state, "web_router", need_search=result.need_search, search_query=result.search_query)
        
    except Exception as e:
        is_timeout = isinstance(e, TimeoutError)
        logger.warning("Web Router %s，使用本地检索结果继续回答。",
                       "调用超时" if is_timeout else "解析失败")
        # 默认值
        result = NetworkSearchResult(need_search=False, search_query="", remain_doc_index=[])
        judge_ai = AIMessage(content=f"解析失败，使用默认值: {result.model_dump()}")
        state["other_messages"].append(judge_ai)
        state.setdefault("node_events", []).append("web_router")
        state["web_router"] = {"need_search": False, "search_query": ""}
        _emit_local_trace(state, "web_router", need_search=False, search_query="",
                          parse_error=not is_timeout, timeout=is_timeout)
    
    # 步骤2：如果需要搜索，执行工具调用
    if result.need_search and result.search_query.strip():
        
        # 创建搜索链
        search_chain = calling_messages | network_search_llm
        
        # Web Router 是额外能力；请求超时或模型未能发起工具调用都不能打断本地 RAG 主链。
        try:
            collector = state.get("local_trace_collector")
            context = state.get("local_trace_context")
            span = (lambda name, **data: collector.span(context, name, **data)) if collector else None
            with (span("web_tool_planner", provider="deepseek", operation="tool_call_plan") if span else nullcontext()):
                search_ai = invoke_with_timeout(
                    lambda: search_chain.invoke({"search_query": result.search_query}),
                    WEB_ROUTER_TIMEOUT_SECONDS,
                )
        except Exception as error:
            error_code = "router_timeout" if "timeout" in type(error).__name__.lower() else "router_error"
            state["web_search_status"] = {
                "called": False,
                "code": error_code,
                "message": "联网搜索路由暂时不可用，本次回答仅基于本地知识库。",
            }
            _emit_local_trace(state, "web_search", called=False, search_query=result.search_query,
                              result_count=0, error_code=error_code)
            logger.warning("联网搜索路由失败：%s；继续使用本地检索结果。", type(error).__name__)
            return state
        state["other_messages"].append(search_ai)
        
        # 检查是否有工具调用
        if _should_call_tool(search_ai):
            collector = state.get("local_trace_collector")
            context = state.get("local_trace_context")
            span = (lambda name, **data: collector.span(context, name, **data)) if collector else nullcontext()
            with span("tavily_search", provider="tavily", operation="search_and_extract"):
                tool_msgs: list[ToolMessage] = network_tool_node.invoke([search_ai])
            state["other_messages"].append(tool_msgs)
            state["web_search_count"] = int(state.get("web_search_count", 0)) + len(tool_msgs)
            state.setdefault("node_events", []).append("web_search")
            web_documents, web_errors, success_count = merge_web_tool_responses(tool_msgs)
            if success_count == 0:
                web_error = web_errors[0] if web_errors else {
                    "code": "invalid_response", "message": "联网搜索未返回可解析结果。",
                }
                state["web_search_status"] = {"called": True, **web_error, "call_count": len(tool_msgs)}
                _emit_local_trace(state, "web_search", called=True, search_query=result.search_query,
                                  result_count=0, call_count=len(tool_msgs), error_code=web_error["code"])
                logger.warning("联网搜索未获得可用结果（%s），继续使用本地检索结果。", web_error["code"])
                return state

            state["web_search_status"] = {
                "called": True,
                "code": "partial_error" if web_errors else None,
                "message": "部分联网搜索失败，已使用其余成功结果。" if web_errors else "",
                "call_count": len(tool_msgs),
                "success_count": success_count,
                "error_count": len(web_errors),
            }
            _emit_local_trace(state, "web_search", called=True, search_query=result.search_query,
                              result_count=len(web_documents), call_count=len(tool_msgs),
                              success_count=success_count, error_count=len(web_errors))

            # 仅在 Web 工具成功返回后更新文档，失败时绝不清空已有本地证据。
            remain_doc = result.remain_doc_index
            if remain_doc:
                valid_indices = [i - 1 for i in remain_doc if 0 < i <= len(state.get("docs", []))]
                state["docs"] = [state["docs"][i] for i in valid_indices]
            else:
                state["docs"] = []
            state["docs"].extend(web_documents)
            if show_debug:
                logger.info("网络检索完成，获得 %s 条文档。", len(web_documents))
        else:
            state["web_search_status"] = {
                "called": False,
                "code": "tool_not_called",
                "message": "联网搜索路由未发起工具调用，本次回答仅基于本地知识库。",
            }
            _emit_local_trace(state, "web_search", called=False, search_query=result.search_query,
                              result_count=0, error_code="tool_not_called")
            logger.warning("联网搜索路由未发起工具调用；继续使用本地检索结果。")
    else:
        if show_debug:
            logger.info(f"信息完整，无需网络搜索...")
    
    return state


def rag(
    state: SearchMessagesState,
    llm: BaseChatModel,
    show_debug: bool
) -> SearchMessagesState:
    if show_debug:
        logger.info(f"开始RAG...")
    sys = get_prompt_template("basic_rag")["system"]
    user = get_prompt_template("basic_rag")["user"]

    prompt = [
        SystemMessage(content=sys),
        HumanMessage(content=user.format(
            all_document_str=format_document_str(state.get("docs", [])),
            input=state["query"]
        ))
    ]
    
    collector = state.get("local_trace_collector")
    context = state.get("local_trace_context")
    with (collector.span(context, "rag_generate", provider="deepseek", operation="generate") if collector else nullcontext()):
        rag_ai = llm.invoke(prompt)
    rag_ai.content = del_think(rag_ai.content)
    if not isinstance(state["main_messages"][-1], AIMessage):
        # 上一轮rag生成合格
        state["main_messages"].append(rag_ai)
    else:
        # 上一轮rag生成不合格，删除上一轮的信息
        state["main_messages"].pop()
        state["main_messages"].append(rag_ai)
    state["summary"] = rag_ai.content
    state.setdefault("node_events", []).append("rag_generate")
    _emit_local_trace(state, "rag_generate", answer_length=len(rag_ai.content or ""))
    return state


def judge(
    state: SearchMessagesState,
    llm: BaseChatModel,
    show_debug: bool
) -> SearchMessagesState:
    """判断节点：负责判断和修改状态"""
    if show_debug:
        logger.info(f"开始评估...")
    collector = state.get("local_trace_collector")
    context = state.get("local_trace_context")
    with (collector.span(context, "judge", provider="deepseek", operation="answer_judge") if collector else nullcontext()):
        judge_ai = llm.invoke([
            SystemMessage(content=get_prompt_template("judge_rag")["system"]),
            HumanMessage(content=get_prompt_template("judge_rag")["user"].format(
                format_document_str=format_document_str(state.get('docs', [])),
                query=state['query'],
                summary=state.get('summary', '')
            ))
        ])
    result = del_think(judge_ai.content or "").strip().lower()
    if show_debug:
        logger.info(f"评估结果{result[:20]}")
    state["other_messages"].append(AIMessage(content=f"[JUDGE]={result}"))
    state.setdefault("node_events", []).append("judge")
    _emit_local_trace(state, "judge", raw_result=result)
    
    # 在这里修改状态
    if 'y' in result: 
        state["judge_result"] = "pass"
    else:
        retries_left = int(state.get("retry", 0)) 
        if retries_left > 0:
            state["retry"] = retries_left - 1  # 状态修改会被保存
            state["judge_retry_count"] = int(state.get("judge_retry_count", 0)) + 1
            state["judge_result"] = "retry"
        else: 
            state["judge_result"] = "fail"
    _emit_local_trace(state, "judge_result", result=state["judge_result"],
                      retry_count=state.get("judge_retry_count", 0))
    
    return state


class SearchGraph:
    def __init__(self, config: AppConfig, power_model: BaseChatModel, websearch_func=tavily_search,
                 retrieval_service: OnlineRetrievalService | None = None) -> None:
        self.config = config
        self.agent_tools = AgentTools(self.config)
        self.agent_tools.register_websearch(websearch_func)
        self.network_search_tool = self.agent_tools.make_web_search_tool()
        
        # bind_tools() 返回新的 RunnableBinding，不修改原模型，无需 deepcopy
        self.network_search_llm = power_model.bind_tools([self.network_search_tool])
        self.llm = create_llm_client(self.config.llm)
        self.retrieval_service = retrieval_service or OnlineRetrievalService(self.config, power_model)
        self.network_tool_node = ToolNode([self.network_search_tool])
        self.search_graph = None

    # ---------- 具有重试回路的图式构建 ----------
    def build_search_graph(self):
        """
        构建图
        """
        def judge_router(state: SearchMessagesState) -> str:
            """简单的路由函数：只读取状态，不修改"""
            return state.get("judge_result", "fail")

        def finish_success(state: SearchMessagesState) -> SearchMessagesState:
            """ 结束节点：成功输出 """
            state["final"] = (state.get("summary", "") or "").strip() or "（空）"
            return state

        def finish_fail(state: SearchMessagesState) -> SearchMessagesState:
            """ 结束节点：失败警告输出 """
            base = (state.get("summary", "") or "").strip() or "（空）"
            state["final"] = base + "\n\n（内容可能不属实）"
            return state
        
        g = StateGraph(SearchMessagesState)

        # 原子节点
        retrieve_node = partial(
            retrieve,
            retrieval_service=self.retrieval_service,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("retrieve", retrieve_node)
        network_search_node = partial(
            llm_network_search,
            judge_llm=self.llm,
            network_search_llm=self.network_search_llm,
            network_tool_node=self.network_tool_node,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("web_search", network_search_node)
        rag_node = partial(
            rag,
            llm=self.llm,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("rag", rag_node)
        g.add_node("finish_success", finish_success)
        g.add_node("finish_fail", finish_fail)
        judge_node = partial(
            judge,
            llm=self.llm,
            show_debug=self.config.multi_dialogue_rag.console_debug
        )
        g.add_node("judge", judge_node)
        # 入口
        g.set_entry_point("retrieve")

        # retrieve -> web_search
        if self.config.agent.network_search_enabled:
            g.add_edge("retrieve", "web_search")
            g.add_edge("web_search", "rag")
        else:
            g.add_edge("retrieve", "rag")
        
        # rag -> judge_router（条件分支）
        # judge -> 条件路由
        if self.config.agent.mode == "analysis":
            g.add_edge("rag", "judge")
            g.add_conditional_edges(
                "judge",  # 从判断节点出发
                judge_router,  # 纯路由函数
                {
                    "pass": "finish_success",
                    "retry": "rag",     
                    "fail": "finish_fail",
                }
            )
            # 结束
            g.add_edge("finish_success", END)
            g.add_edge("finish_fail", END)
        elif self.config.agent.mode == "fast":
            g.add_edge("rag", END)

        self.search_graph = g.compile()

    # ---------- 对外：跑整张图，返回最终输出 ----------
    def answer(self, query: str) -> str:
        if self.search_graph is None:
            self.build_search_graph()
        init_state: SearchMessagesState = {
            "query": query,
            "main_messages": [HumanMessage(content=query)],
            "other_messages": [],
            "docs": [],
            "summary": "",
            "retry": self.config.agent.max_attempts,
            "final": "",
            "retrieval_info": {},
            "web_search_count": 0,
            "web_search_status": {},
            "judge_retry_count": 0,
            "node_events": [],
        }
        # 执行图
        out_state: SearchMessagesState = self.search_graph.invoke(init_state, {"run_name": "search_graph"})
        return out_state.get("final", "") or out_state.get("summary", "") or "（空）"
    
    def run(self, init_state: SearchMessagesState) -> SearchMessagesState:
        if self.search_graph is None:
            self.build_search_graph()
        out_state: SearchMessagesState = self.search_graph.invoke(init_state, {"run_name": "search_graph"})
        return out_state
