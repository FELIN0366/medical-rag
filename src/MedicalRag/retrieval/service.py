"""Stage 4 在线检索服务：复用 Stage 2 的规划、执行和重排主链。"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from ..config.models import AppConfig
from ..core.DBFactory import get_kb
from .executor import UnifiedRetrievalExecutor
from .policy import AgentPlannerPolicy
from .reranker import DashScopeReranker


@dataclass(frozen=True)
class OnlineRetrievalResult:
    """一次真实在线检索的 Top-5 结果及可审计执行元数据。"""

    documents: list[Document]
    selected_channels: tuple[str, ...]
    planner_error: str | None
    candidate_count: int
    planner_latency_ms: float
    retrieval_latency_ms: float
    reranker_latency_ms: float


class OnlineRetrievalService:
    """唯一在线 Retrieval 编排层，不复制 Stage 2 的算法实现。"""

    def __init__(
        self,
        config: AppConfig,
        planner_llm: BaseChatModel,
        *,
        planner: AgentPlannerPolicy | None = None,
        executor: UnifiedRetrievalExecutor | None = None,
        reranker: DashScopeReranker | None = None,
    ) -> None:
        self.config = config
        self.planner = planner or AgentPlannerPolicy(config, planner_llm)
        self.executor = executor or UnifiedRetrievalExecutor(get_kb(config.model_dump()), config)
        self.reranker = reranker or DashScopeReranker(config.reranker)

    async def retrieve_async(self, query: str, *, trace_collector: Any = None,
                             trace_context: dict | None = None) -> OnlineRetrievalResult:
        """异步入口供评测或异步宿主复用，重排实现仅维护一份。"""
        span = (lambda name, **data: trace_collector.span(trace_context, name, **data)) if trace_collector else None
        with (span("retrieval_planner", provider="deepseek", operation="tool_plan") if span else nullcontext()):
            plan = self.planner.plan(query)
        with (span("retrieval_executor", provider="dashscope", operation="embedding_and_milvus") if span else nullcontext()):
            trace = self.executor.execute(plan)
        candidates = trace.documents[: self.config.retrieval_stage2.candidate_k]
        started = time.perf_counter()
        with (span("reranker", provider="dashscope", operation="rerank", model=self.config.reranker.model) if span else nullcontext()):
            ranked = await self.reranker.rerank(
                query,
                [document.page_content for document in candidates],
                self.config.retrieval_stage2.final_k,
            )
        reranker_latency_ms = (time.perf_counter() - started) * 1000
        by_index = {item.index: item.score for item in ranked}
        ordered_indices = sorted(by_index, key=lambda index: (-by_index[index], index))
        documents = [candidates[index] for index in ordered_indices[: self.config.retrieval_stage2.final_k]]
        return OnlineRetrievalResult(
            documents=documents,
            selected_channels=plan.selected_channels,
            planner_error=plan.planner_error,
            candidate_count=len(candidates),
            planner_latency_ms=float(getattr(self.planner, "last_planner_latency_ms", 0.0)),
            retrieval_latency_ms=trace.retrieval_latency_ms,
            reranker_latency_ms=reranker_latency_ms,
        )

    def retrieve(self, query: str, *, trace_collector: Any = None,
                 trace_context: dict | None = None) -> OnlineRetrievalResult:
        """供同步 SearchGraph 使用的薄入口。运行中的事件循环应使用异步入口。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.retrieve_async(query, trace_collector=trace_collector, trace_context=trace_context))
        raise RuntimeError("同步检索入口不能在运行中的事件循环调用，请使用 retrieve_async")
