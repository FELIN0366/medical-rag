from typing import List, Dict, TypedDict
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from ...config.models import AppConfig
from langchain.tools import tool
from ...core.KnowledgeBase import MedicalHybridKnowledgeBase
from ...config.models import SearchRequest
from langchain_core.documents import Document
import ast
import operator as op
from langchain.tools import tool
import json
import logging
from threading import Thread
from ...core.DBFactory import get_kb

logger = logging.getLogger(__name__)

WEB_SEARCH_TIMEOUT_SECONDS = 60

class AgentTools:
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        self.WEBSEARCH_FUNC = None
        
    def register_websearch(self, func):
        self.WEBSEARCH_FUNC = func
        
    def make_web_search_tool(self):
        if self.WEBSEARCH_FUNC is None:
            raise RuntimeError("未注册网络检索工具")
        
        cnt = self.app_config.agent.network_search_cnt
        
        @tool("web_search")
        def web_search(query: str) -> str:
            """使用输入文本进行联网搜索，返回包含 documents 与 error 的 JSON 对象。"""
            result_box = []
            error_box = []

            def invoke_provider() -> None:
                try:
                    result_box.append(self.WEBSEARCH_FUNC(query, cnt))
                except Exception as error:
                    error_box.append(error)

            # Tavily SDK 的远端提取可能没有在传入 timeout 后及时返回；
            # 这里使用 daemon 线程提供调用方可观察、可降级的硬上界。
            worker = Thread(target=invoke_provider, daemon=True)
            worker.start()
            worker.join(WEB_SEARCH_TIMEOUT_SECONDS)
            if worker.is_alive():
                logger.warning("联网搜索调用超时（%s 秒）", WEB_SEARCH_TIMEOUT_SECONDS)
                return json.dumps({
                    "ok": False,
                    "documents": [],
                    "error": {
                        "code": "provider_timeout",
                        "message": "联网搜索服务响应超时，本次回答仅基于本地知识库。",
                    },
                }, ensure_ascii=False)

            if error_box:
                error = error_box[0]
                if isinstance(error, KeyError):
                    # 不把原始异常交给模型；避免 ToolNode 将异常转成非 JSON 文本。
                    missing_name = str(error).strip("'")
                    logger.warning("联网搜索未执行：缺少环境变量 %s", missing_name)
                    return json.dumps({
                        "ok": False,
                        "documents": [],
                        "error": {
                            "code": "configuration_error",
                            "message": f"联网搜索服务未配置：缺少 {missing_name}。",
                        },
                    }, ensure_ascii=False)
                # Provider 的网络、认证、限流等错误均保留为受控的工具结果，不能中断 RAG 主链。
                logger.warning("联网搜索调用失败：%s", type(error).__name__)
                return json.dumps({
                    "ok": False,
                    "documents": [],
                    "error": {
                        "code": "provider_error",
                        "message": "联网搜索服务暂时不可用，本次回答仅基于本地知识库。",
                    },
                }, ensure_ascii=False)

            try:
                results: List[Document] = result_box[0]
                documents = [document.model_dump() for document in results]
            except Exception as error:
                logger.warning("联网搜索返回内容无效：%s", type(error).__name__)
                return json.dumps({
                    "ok": False,
                    "documents": [],
                    "error": {
                        "code": "invalid_provider_result",
                        "message": "联网搜索服务暂时不可用，本次回答仅基于本地知识库。",
                    },
                }, ensure_ascii=False)

            return json.dumps({
                "ok": True,
                "documents": documents,
                "error": None,
            }, ensure_ascii=False)
        
        return web_search
    
    def make_database_search_tool(self):
        
        @tool("database_search")
        def database_search(search_config: SearchRequest) -> str:
            """ 输入检索参数,使用向量数据库进行本地检索 """
            results = get_kb(self.app_config.model_dump()).search(req=search_config)
            return json.dumps([d.model_dump() for d in results], ensure_ascii=False)
        
        return database_search
    
    def make_calculator_tool(self):
        # 支持的运算符
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.USub: op.neg,
        }
        
        def eval_expr(expr: str) -> float:
            """安全地计算表达式"""
            def _eval(node):
                if isinstance(node, ast.Num):  # 处理常数
                    return node.n
                elif isinstance(node, ast.BinOp):  # 二元运算
                    return operators[type(node.op)](_eval(node.left), _eval(node.right))
                elif isinstance(node, ast.UnaryOp):  # 负号等
                    return operators[type(node.op)](_eval(node.operand))
                else:
                    raise ValueError("不支持的表达式")
            
            node = ast.parse(expr, mode="eval").body
            return _eval(node)
        
        @tool("calculator")
        def calculator(expression: str) -> str:
            """计算器：输入一个算术表达式字符串，返回结果。支持 + - * /"""
            try:
                result = eval_expr(expression)
                return str(result)
            except Exception as e:
                return f"计算出错: {e}"
        
        return calculator
