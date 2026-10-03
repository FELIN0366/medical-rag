import asyncio

import httpx
from langchain_core.documents import Document

from MedicalRag.config.models import AppConfig, EmbeddingConfig, MilvusConfig, RerankerConfig, SearchRequest
from MedicalRag.evaluation.retrieval_dataset import EvalRecord
from MedicalRag.evaluation.retrieval_runner import RetrievalEvaluationRunner
from MedicalRag.retrieval.executor import UnifiedRetrievalExecutor
from MedicalRag.retrieval.policy import FixedHybridPolicy, normalize_plan
from MedicalRag.retrieval.reranker import DashScopeReranker


def _config() -> AppConfig:
    return AppConfig(milvus=MilvusConfig(), embedding=EmbeddingConfig())


def test_fixed_policy_freezes_three_routes_and_candidate_limit() -> None:
    config = _config()
    plan = FixedHybridPolicy(config).plan("高血压防治")
    assert plan.selected_channels == ("summary_dense", "text_dense", "text_sparse")
    assert plan.search_request.limit == 30
    assert [request.limit for request in plan.search_request.requests] == [50, 50, 50]
    assert plan.search_request.fuse is not None
    assert plan.search_request.fuse.k == 60
    assert "pk" in plan.search_request.output_fields


def test_default_search_request_includes_stable_primary_key() -> None:
    assert "pk" in SearchRequest().output_fields


def test_normalize_plan_ignores_unapproved_planner_parameters() -> None:
    plan = normalize_plan("阿司匹林禁忌症", ("text_sparse", "invalid", "summary_dense"), _config(),
                          policy_name="agent_planner")
    assert plan.selected_channels == ("text_sparse", "summary_dense")
    assert plan.search_request.requests[0].metric_type == "BM25"
    assert plan.search_request.requests[1].search_params == {"ef": 64}


class _FakeKnowledgeBase:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode_query_vectors(self, query, fields):
        self.encode_calls += 1
        return {field: query if field == "text_sparse" else [1.0, 0.0] for field in fields}

    def search(self, request, encoded_queries):
        assert set(encoded_queries) == {item.anns_field for item in request.requests}
        return [Document(page_content="候选", metadata={"pk": "p1"})]


def test_executor_reuses_query_embeddings_across_raw_routes() -> None:
    config = _config()
    knowledge_base = _FakeKnowledgeBase()
    executor = UnifiedRetrievalExecutor(knowledge_base, config)
    trace = executor.raw_routes("高血压防治")
    assert set(trace.raw_routes) == {"summary_dense", "text_dense", "text_sparse"}
    # 三路预热时 summary/text 使用同一次真实 embedding 调用，后续原子路由只读缓存。
    assert knowledge_base.encode_calls == 1


def test_prepare_queries_batches_by_twenty_and_shares_dense_vectors() -> None:
    class FakeEmbeddings:
        def __init__(self) -> None:
            self.batches = []

        def embed_documents(self, texts):
            self.batches.append(list(texts))
            return [[float(index)] for index, _ in enumerate(texts)]

    class BatchKnowledgeBase(_FakeKnowledgeBase):
        def __init__(self) -> None:
            super().__init__()
            self.summary_embedding = FakeEmbeddings()
            self.text_embedding = FakeEmbeddings()

    knowledge_base = BatchKnowledgeBase()
    executor = UnifiedRetrievalExecutor(knowledge_base, _config())
    queries = [f"查询-{index}" for index in range(41)] + ["查询-0"]
    executor.prepare_queries(queries, batch_size=20)
    assert [len(batch) for batch in knowledge_base.summary_embedding.batches] == [20, 20, 1]
    assert knowledge_base.text_embedding.batches == []
    cached = executor.cached_query_vectors()
    assert len(cached) == 41
    assert cached["查询-0"]["summary_dense"] is cached["查询-0"]["text_dense"]


def test_reranker_sends_one_document_list_request(monkeypatch) -> None:
    received = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={"output": {"results": [
            {"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4},
        ]}})

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "workspace")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = DashScopeReranker(RerankerConfig(), client=client)
    results = asyncio.run(reranker.rerank("问题", ["候选一", "候选二"]))
    asyncio.run(client.aclose())
    assert [(item.index, item.score) for item in results] == [(1, 0.9), (0, 0.4)]
    assert len(received) == 1
    payload = __import__("json").loads(received[0].content)
    assert payload["input"]["documents"] == ["候选一", "候选二"]


def test_runner_sends_document_page_content_to_reranker() -> None:
    class FakeReranker:
        async def rerank(self, query, documents, top_n):
            self.query = query
            self.documents = documents
            self.top_n = top_n
            return []

    runner = object.__new__(RetrievalEvaluationRunner)
    runner.config = _config()
    runner.reranker = FakeReranker()
    record = EvalRecord(query_id="q", query="查询", dataset="qa", relevant_pks=["p1"])
    candidates = [Document(page_content="冻结 chunk text", metadata={"pk": "p1", "document": "不得发送该字段"})]
    ordered, _, _ = asyncio.run(runner._rerank(record, candidates))
    assert runner.reranker.documents == ["冻结 chunk text"]
    assert ordered == candidates
