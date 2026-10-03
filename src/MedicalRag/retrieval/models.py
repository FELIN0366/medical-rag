"""Stage 2 主链使用的轻量运行时模型。"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from ..config.models import SearchRequest


@dataclass(frozen=True)
class RetrievalPlan:
    """已规范化的检索计划，Planner 只可影响 selected_channels。"""
    policy_name: str
    selected_channels: tuple[str, ...]
    search_request: SearchRequest
    planner_error: str | None = None


@dataclass
class RetrievalTrace:
    """一次候选召回的可序列化执行信息。"""
    plan: RetrievalPlan
    documents: list[Document]
    query_embedding_reused: bool
    retrieval_latency_ms: float
    raw_routes: dict[str, list[Document]] = field(default_factory=dict)
