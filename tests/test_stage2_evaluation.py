"""Stage 2 Eval 数据、指标与批量缓存的无网络回归测试。"""
from __future__ import annotations

import asyncio
import json

from langchain_core.documents import Document

from MedicalRag.config.models import AppConfig, EmbeddingConfig, MilvusConfig
from MedicalRag.evaluation.retrieval_dataset import EvalRecord, sample_qa_rows, write_eval_records
from MedicalRag.evaluation.retrieval_metrics import evaluate_predictions, ndcg_at_k
from MedicalRag.evaluation.retrieval_runner import RetrievalEvaluationRunner
from MedicalRag.retrieval.models import RetrievalPlan
from MedicalRag.retrieval.policy import normalize_plan
from MedicalRag.retrieval.reranker import RerankResult


def test_metrics_use_candidate_pool_for_recall_and_final_for_ranking() -> None:
    record = EvalRecord(query_id="q1", query="测试", dataset="qa", relevant_pks=["gold"],
                        partial_relevant_pks=["partial"])
    prediction = {
        "query_id": "q1",
        "candidate_pks": ["x", "gold"],
        "final_pks": ["partial", "x"],
    }
    report = evaluate_predictions([record], [prediction])["qa"]
    assert report["candidate_recall_at_30"] == 1.0
    assert report["recall_at_5"] == 0.0
    assert report["mrr"] == 0.0
    assert ndcg_at_k(record, prediction, 5) > 0.0


def test_fixed_qa_sampling_is_independent_of_input_order() -> None:
    rows = [{"pk": f"pk-{index:03d}", "summary": str(index)} for index in range(40)]
    forward = sample_qa_rows(rows, seed=42, size=30)
    backward = sample_qa_rows(list(reversed(rows)), seed=42, size=30)
    assert [row["pk"] for row in forward] == [row["pk"] for row in backward]


def test_eval_record_writer_emits_utf8_jsonl(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    write_eval_records(path, [EvalRecord(query_id="q", query="中文问题", dataset="qa")])
    assert json.loads(path.read_text(encoding="utf-8"))["query"] == "中文问题"


def test_runner_writes_all_runtime_artifacts_with_one_executor(tmp_path) -> None:
    config = AppConfig(milvus=MilvusConfig(), embedding=EmbeddingConfig())

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]

    class FakeKnowledgeBase:
        summary_embedding = FakeEmbeddings()
        text_embedding = summary_embedding

        def search(self, request, encoded_queries):
            assert set(encoded_queries) == {item.anns_field for item in request.requests}
            channels = {item.anns_field for item in request.requests}
            if channels == {"summary_dense"}:
                pks = ["p1", "p2"]
            elif channels == {"text_dense"}:
                pks = ["p2", "p1"]
            elif channels == {"text_sparse"}:
                pks = ["p2", "p3"]
            else:
                pks = ["p2", "p1", "p3"]
            return [Document(page_content=f"文本 {pk}", metadata={"pk": pk, "source": "qa"}) for pk in pks]

    class FakeReranker:
        async def rerank(self, query, documents, top_n):
            assert len(documents) == top_n
            return [RerankResult(index=index, score=float(index)) for index in range(len(documents))]

    class FakeAgentPolicy:
        last_planner_latency_ms = 1.25

        def plan(self, query):
            return normalize_plan(query, ("summary_dense",), config, policy_name="agent_planner")

    runner = RetrievalEvaluationRunner(
        config,
        knowledge_base=FakeKnowledgeBase(),
        reranker=FakeReranker(),
        agent_policy=FakeAgentPolicy(),
    )
    record = EvalRecord(query_id="qa-01", query="测试问题", dataset="qa", relevant_pks=["p1"])
    asyncio.run(runner.run([record], tmp_path))
    expected = {
        "query_embeddings.npz", "query_embeddings_meta.json", "raw_routes.jsonl",
        "fixed_hybrid_predictions.jsonl", "fixed_rerank_predictions.jsonl", "agent_rerank_predictions.jsonl",
        "metrics.json", "reranker_analysis.json", "planner_analysis.json", "failure_cases.jsonl",
        "latency.json", "config_snapshot.yaml",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    categories = [json.loads(line)["category"] for line in (tmp_path / "failure_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert categories == ["reranker_rescue", "reranker_damage", "planner_win", "planner_loss", "bm25_only_strong", "dense_strong"]
    assert set(json.loads((tmp_path / "reranker_analysis.json").read_text(encoding="utf-8"))) == {"qa"}
    assert set(json.loads((tmp_path / "planner_analysis.json").read_text(encoding="utf-8"))) == {"qa"}
