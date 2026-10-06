"""Stage 2 的固定策略与复用既有 Tool Schema 的 Agent 选路策略。"""
from __future__ import annotations

import time
from typing import Callable, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..agent.tools.AgentTools import AgentTools
from ..config.models import AppConfig, FusionSpec, SearchRequest, SingleSearchRequest
from ..prompts.templates import get_prompt_template
from .models import RetrievalPlan

CHANNELS = ("summary_dense", "text_dense", "text_sparse")


def normalize_plan(query: str, selected_channels: Iterable[str], config: AppConfig,
                   *, policy_name: str, planner_error: str | None = None) -> RetrievalPlan:
    """忽略 Planner 的数据库参数，仅保留其选中的有效通道。"""
    chosen = tuple(dict.fromkeys(field for field in selected_channels if field in CHANNELS))
    if not chosen:
        raise ValueError("检索计划至少必须选择一个有效通道")
    frozen = config.retrieval_stage2
    requests: list[SingleSearchRequest] = []
    for field in chosen:
        if field == "text_sparse":
            requests.append(SingleSearchRequest(anns_field=field, metric_type="BM25",
                                                search_params={"drop_ratio_search": 0.0}, limit=frozen.per_route_k))
        else:
            requests.append(SingleSearchRequest(anns_field=field, metric_type="COSINE",
                                                search_params={"ef": 64}, limit=frozen.per_route_k))
    # 在线运行时与评测使用同一冻结检索参数；此处只补齐下游生成和追踪所需的证据定位字段。
    request = SearchRequest(
        query=query,
        collection_name=config.milvus.collection_name,
        requests=requests,
        output_fields=[
            "pk", "text", "summary", "document", "source", "source_name", "doc_id", "chunk_id",
            "department", "title", "section_path", "page",
        ],
        fuse=FusionSpec(method="rrf", k=frozen.rrf_k) if len(requests) > 1 else None,
        limit=frozen.candidate_k,
    )
    return RetrievalPlan(policy_name=policy_name, selected_channels=chosen, search_request=request,
                         planner_error=planner_error)


class FixedHybridPolicy:
    """固定三路 S+T+B 的强基线策略。"""
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def plan(self, query: str) -> RetrievalPlan:
        return normalize_plan(query, CHANNELS, self.config, policy_name="fixed_hybrid")


class AgentPlannerPolicy:
    """基于既有 database_search Tool Schema 的 Stage 2 选路策略。"""
    def __init__(self, config: AppConfig, planner_llm: BaseChatModel,
                 proposal_override: Callable[[str], Iterable[str]] | None = None) -> None:
        self.config = config
        self.planner_llm = planner_llm
        self.proposal_override = proposal_override
        # 与现有 SearchGraph 使用相同的结构化检索工具 Schema；这里仅解析计划，不执行 ToolNode。
        self.database_search_tool = AgentTools(config).make_database_search_tool()
        self.bound_llm = planner_llm.bind_tools([self.database_search_tool])
        self.last_planner_latency_ms = 0.0

    def _tool_selected_channels(self, query: str) -> tuple[tuple[str, ...], str | None]:
        started = time.perf_counter()
        try:
            template = get_prompt_template("call_db")
            response = self.bound_llm.invoke([
                SystemMessage(content=template["system"]),
                HumanMessage(content=template["user"].format(query=query)),
            ])
            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                return (), "Planner 未返回 database_search Tool Call"
            arguments = calls[0].get("args", {})
            payload = arguments.get("search_config", arguments)
            proposal = SearchRequest.model_validate(payload)
            return tuple(item.anns_field for item in proposal.requests), None
        except Exception as exc:
            return (), f"Planner 计划解析失败：{type(exc).__name__}: {exc}"
        finally:
            self.last_planner_latency_ms = (time.perf_counter() - started) * 1000

    def plan(self, query: str) -> RetrievalPlan:
        if self.proposal_override is not None:
            selected = tuple(self.proposal_override(query))
            return normalize_plan(query, selected, self.config, policy_name="agent_planner")
        selected, error = self._tool_selected_channels(query)
        # Planner 服务异常时退回固定三路，并显式记录，避免把失败伪装成动态选路。
        if not selected:
            return normalize_plan(query, CHANNELS, self.config, policy_name="agent_planner", planner_error=error)
        return normalize_plan(query, selected, self.config, policy_name="agent_planner", planner_error=error)
