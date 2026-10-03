"""Fixed 与 Agent Policy 共用的唯一 Milvus 检索执行器。"""
from __future__ import annotations

import time
from typing import Iterable

from ..config.models import AppConfig
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from .models import RetrievalPlan, RetrievalTrace
from .policy import normalize_plan


class UnifiedRetrievalExecutor:
    """统一执行规范化计划，并复用同一 query 的 dense 向量。"""
    def __init__(self, knowledge_base: MedicalHybridKnowledgeBase, config: AppConfig) -> None:
        self.knowledge_base = knowledge_base
        self.config = config
        self._query_vectors: dict[str, dict[str, list[float] | str]] = {}

    def clear_query_cache(self) -> None:
        self._query_vectors.clear()

    def prepare_queries(self, queries: Iterable[str], batch_size: int = 20) -> None:
        """预先按批缓存 Eval 查询向量，避免 A/B/C 重复调用 Embedding 服务。"""
        unique_queries = list(dict.fromkeys(query for query in queries if query))
        dense_same = (
            self.config.embedding.summary_dense.model_dump()
            == self.config.embedding.text_dense.model_dump()
        )
        for start in range(0, len(unique_queries), batch_size):
            batch = [query for query in unique_queries[start:start + batch_size] if query not in self._query_vectors]
            if not batch:
                continue
            if dense_same:
                vectors = self.knowledge_base.summary_embedding.embed_documents(batch)
                if len(vectors) != len(batch):
                    raise RuntimeError("批量查询 Embedding 返回数量不匹配")
                for query, vector in zip(batch, vectors):
                    self._query_vectors[query] = {
                        "summary_dense": vector,
                        "text_dense": vector,
                        "text_sparse": query,
                    }
            else:
                summary_vectors = self.knowledge_base.summary_embedding.embed_documents(batch)
                text_vectors = self.knowledge_base.text_embedding.embed_documents(batch)
                if len(summary_vectors) != len(batch) or len(text_vectors) != len(batch):
                    raise RuntimeError("批量查询 Embedding 返回数量不匹配")
                for query, summary, text in zip(batch, summary_vectors, text_vectors):
                    self._query_vectors[query] = {
                        "summary_dense": summary,
                        "text_dense": text,
                        "text_sparse": query,
                    }

    def cached_query_vectors(self) -> dict[str, dict[str, list[float] | str]]:
        """返回副本，供评测脚本写入 NPZ，不暴露内部可变缓存。"""
        return {query: dict(vectors) for query, vectors in self._query_vectors.items()}

    def _vectors(self, query: str, fields: Iterable[str]) -> tuple[dict[str, list[float] | str], bool]:
        requested = set(fields)
        cached = self._query_vectors.setdefault(query, {})
        missing = requested - set(cached)
        if missing:
            cached.update(self.knowledge_base.encode_query_vectors(query, missing))
        return {field: cached[field] for field in requested}, not bool(missing)

    def execute(self, plan: RetrievalPlan) -> RetrievalTrace:
        vectors, reused = self._vectors(plan.search_request.query, plan.selected_channels)
        started = time.perf_counter()
        documents = self.knowledge_base.search(plan.search_request, encoded_queries=vectors)
        return RetrievalTrace(plan=plan, documents=documents, query_embedding_reused=reused,
                              retrieval_latency_ms=(time.perf_counter() - started) * 1000)

    def raw_routes(self, query: str) -> RetrievalTrace:
        """Evaluation 模式预跑三路 Top-50，缓存仅用于分析而非 Planner 输入。"""
        # 预热全部通道，令 summary/text 相同的 embedding 配置在一次 API 调用中复用。
        self._vectors(query, ("summary_dense", "text_dense", "text_sparse"))
        route_docs = {}
        total_latency = 0.0
        for channel in ("summary_dense", "text_dense", "text_sparse"):
            plan = normalize_plan(query, (channel,), self.config, policy_name="raw_route")
            trace = self.execute(plan)
            route_docs[channel] = trace.documents
            total_latency += trace.retrieval_latency_ms
        all_plan = normalize_plan(query, ("summary_dense", "text_dense", "text_sparse"), self.config,
                                  policy_name="raw_routes")
        return RetrievalTrace(plan=all_plan, documents=[], query_embedding_reused=True,
                              retrieval_latency_ms=total_latency, raw_routes=route_docs)
