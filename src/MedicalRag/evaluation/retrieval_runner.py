"""Stage 2 唯一评测运行器：Fixed、Rerank 与 Agent 共用检索和重排链。"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from langchain_core.documents import Document

from ..config.models import AppConfig
from ..core.KnowledgeBase import MedicalHybridKnowledgeBase
from ..core.utils import create_llm_client
from ..retrieval.executor import UnifiedRetrievalExecutor
from ..retrieval.policy import AgentPlannerPolicy, FixedHybridPolicy
from ..retrieval.reranker import DashScopeReranker
from .retrieval_dataset import EvalRecord
from .retrieval_metrics import evaluate_predictions


def _document_item(document: Document) -> dict:
    """将 LangChain Document 缩为稳定、可审计的预测记录。"""
    metadata = document.metadata
    return {
        "pk": str(metadata.get("pk", "")),
        "source": metadata.get("source", ""),
        "source_name": metadata.get("source_name", ""),
        "title": metadata.get("title", ""),
        "section_path": metadata.get("section_path", ""),
        "distance": metadata.get("distance", 0.0),
        "text": document.page_content,
        "document": metadata.get("document", ""),
    }


def _rank_of_gold(pks: Iterable[str], record: EvalRecord) -> int | None:
    gold = record.direct_gold_pks()
    for rank, pk in enumerate(pks, start=1):
        if pk in gold:
            return rank
    return None


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "count": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
    }


class RetrievalEvaluationRunner:
    """严格执行 Exp A/B/C；不写入或修改 Milvus collection。"""
    def __init__(self, config: AppConfig, knowledge_base: MedicalHybridKnowledgeBase | None = None,
                 reranker: DashScopeReranker | None = None,
                 agent_policy: AgentPlannerPolicy | None = None) -> None:
        self.config = config
        self.knowledge_base = knowledge_base or MedicalHybridKnowledgeBase(config)
        self.executor = UnifiedRetrievalExecutor(self.knowledge_base, config)
        self.fixed_policy = FixedHybridPolicy(config)
        self.agent_policy = agent_policy or AgentPlannerPolicy(config, create_llm_client(config.llm))
        self.reranker = reranker or DashScopeReranker(config.reranker)

    async def _rerank(self, record: EvalRecord, candidates: list[Document]) -> tuple[list[Document], float, list[dict]]:
        """每条 query 仅向 Reranker 发送一次完整 Candidate Top-30。"""
        if len(candidates) > self.config.retrieval_stage2.candidate_k:
            raise RuntimeError("统一执行器返回的候选数超过 Stage 2 Candidate Top-30")
        started = time.perf_counter()
        results = await self.reranker.rerank(
            record.query,
            [document.page_content for document in candidates],
            top_n=len(candidates),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        scores = {item.index: item.score for item in results}
        ordered_indices = sorted(range(len(candidates)), key=lambda index: scores.get(index, float("-inf")), reverse=True)
        ordered = [candidates[index] for index in ordered_indices]
        audit = [{"candidate_index": index, "pk": str(candidates[index].metadata.get("pk", "")),
                  "relevance_score": scores.get(index)} for index in ordered_indices]
        return ordered, latency_ms, audit

    def _prediction(self, record: EvalRecord, *, experiment: str, candidates: list[Document], final: list[Document],
                    selected_channels: tuple[str, ...], retrieval_latency_ms: float, planner_latency_ms: float = 0.0,
                    rerank_latency_ms: float = 0.0, rerank_audit: list[dict] | None = None,
                    planner_error: str | None = None, reranked_pks: list[str] | None = None) -> dict:
        return {
            "query_id": record.query_id,
            "dataset": record.dataset,
            "query": record.query,
            "experiment": experiment,
            "selected_channels": list(selected_channels),
            "candidate_pks": [str(document.metadata.get("pk", "")) for document in candidates],
            "final_pks": [str(document.metadata.get("pk", "")) for document in final],
            "reranked_pks": reranked_pks or [],
            "candidate_documents": [_document_item(document) for document in candidates],
            "final_documents": [_document_item(document) for document in final],
            "planner_error": planner_error,
            "latency_ms": {
                "planner": planner_latency_ms,
                "retrieval": retrieval_latency_ms,
                "reranker": rerank_latency_ms,
            },
            "rerank_results": rerank_audit or [],
        }

    async def run(self, records: list[EvalRecord], output_dir: Path) -> dict:
        """执行三组实验并原子地生成 Stage 2 的运行产物。"""
        if not records:
            raise RuntimeError("没有可执行的 Eval Record")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.executor.prepare_queries([record.query for record in records], batch_size=20)
        self._write_embedding_cache(output_dir)

        fixed_predictions: list[dict] = []
        rerank_predictions: list[dict] = []
        agent_predictions: list[dict] = []
        raw_route_rows: list[dict] = []
        for record in records:
            raw_trace = self.executor.raw_routes(record.query)
            raw_route_rows.append({
                "query_id": record.query_id,
                "dataset": record.dataset,
                "query": record.query,
                "routes": {channel: [_document_item(document) for document in documents]
                           for channel, documents in raw_trace.raw_routes.items()},
            })

            fixed_trace = self.executor.execute(self.fixed_policy.plan(record.query))
            fixed_final = fixed_trace.documents[:self.config.retrieval_stage2.final_k]
            fixed_predictions.append(self._prediction(
                record, experiment="fixed_hybrid", candidates=fixed_trace.documents, final=fixed_final,
                selected_channels=fixed_trace.plan.selected_channels, retrieval_latency_ms=fixed_trace.retrieval_latency_ms,
            ))

            reranked_all, rerank_latency, audit = await self._rerank(record, fixed_trace.documents)
            reranked_final = reranked_all[:self.config.retrieval_stage2.final_k]
            rerank_predictions.append(self._prediction(
                record, experiment="fixed_rerank", candidates=fixed_trace.documents, final=reranked_final,
                selected_channels=fixed_trace.plan.selected_channels, retrieval_latency_ms=fixed_trace.retrieval_latency_ms,
                rerank_latency_ms=rerank_latency, rerank_audit=audit,
                reranked_pks=[str(document.metadata.get("pk", "")) for document in reranked_all],
            ))

            agent_plan = self.agent_policy.plan(record.query)
            agent_trace = self.executor.execute(agent_plan)  # 线上模式只执行 Planner 实际选择的 route。
            agent_reranked_all, agent_rerank_latency, agent_audit = await self._rerank(record, agent_trace.documents)
            agent_final = agent_reranked_all[:self.config.retrieval_stage2.final_k]
            agent_predictions.append(self._prediction(
                record, experiment="agent_rerank", candidates=agent_trace.documents, final=agent_final,
                selected_channels=agent_trace.plan.selected_channels, retrieval_latency_ms=agent_trace.retrieval_latency_ms,
                planner_latency_ms=self.agent_policy.last_planner_latency_ms,
                rerank_latency_ms=agent_rerank_latency, rerank_audit=agent_audit,
                planner_error=agent_plan.planner_error,
                reranked_pks=[str(document.metadata.get("pk", "")) for document in agent_reranked_all],
            ))

        self._write_jsonl(output_dir / "raw_routes.jsonl", raw_route_rows)
        self._write_jsonl(output_dir / "fixed_hybrid_predictions.jsonl", fixed_predictions)
        self._write_jsonl(output_dir / "fixed_rerank_predictions.jsonl", rerank_predictions)
        self._write_jsonl(output_dir / "agent_rerank_predictions.jsonl", agent_predictions)
        report = self._write_reports(records, fixed_predictions, rerank_predictions, agent_predictions, raw_route_rows, output_dir)
        self._write_config_snapshot(output_dir)
        return report

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_embedding_cache(self, output_dir: Path) -> None:
        vectors = self.executor.cached_query_vectors()
        dense = {query: value["summary_dense"] for query, value in vectors.items()}
        np.savez_compressed(output_dir / "query_embeddings.npz", **dense)
        (output_dir / "query_embeddings_meta.json").write_text(json.dumps({
            "model": self.config.embedding.summary_dense.model,
            "dimension": self.config.embedding.summary_dense.dimension,
            "batch_size": 20,
            "query_count": len(dense),
            "queries": list(dense),
            "shared_summary_text_vector": (
                self.config.embedding.summary_dense.model_dump() == self.config.embedding.text_dense.model_dump()
            ),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _best_rank(record: EvalRecord, prediction: dict) -> int | None:
        return _rank_of_gold(prediction["final_pks"], record)

    def _write_reports(self, records: list[EvalRecord], fixed: list[dict], rerank: list[dict], agent: list[dict],
                       raw_routes: list[dict], output_dir: Path) -> dict:
        fixed_by_id = {item["query_id"]: item for item in fixed}
        rerank_by_id = {item["query_id"]: item for item in rerank}
        agent_by_id = {item["query_id"]: item for item in agent}
        metrics = {
            "fixed_hybrid": evaluate_predictions(records, fixed),
            "fixed_rerank": evaluate_predictions(records, rerank),
            "agent_rerank": evaluate_predictions(records, agent),
        }
        metrics["fixed_hybrid"]["config"] = "S+T+B / Top-50 each / RRF(60) / Candidate Top-30 / Final Top-5"
        metrics["fixed_rerank"]["config"] = "Fixed Hybrid Candidate Top-30 + qwen3.7-text-rerank Top-5"
        metrics["agent_rerank"]["config"] = "Planner selected subset + same Reranker Top-5"
        (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        records_by_dataset: dict[str, list[EvalRecord]] = defaultdict(list)
        for record in records:
            records_by_dataset[record.dataset].append(record)

        rerank_report: dict[str, dict] = {}
        planner_report: dict[str, dict] = {}
        for dataset, items in records_by_dataset.items():
            rescues, damages, rank_gains = [], [], []
            for record in items:
                before = self._best_rank(record, fixed_by_id[record.query_id])
                after = self._best_rank(record, rerank_by_id[record.query_id])
                if before is None and after is not None:
                    rescues.append(record.query_id)
                if before is not None and after is None:
                    damages.append(record.query_id)
                candidate_rank = _rank_of_gold(fixed_by_id[record.query_id]["candidate_pks"], record)
                rerank_rank = _rank_of_gold(rerank_by_id[record.query_id]["reranked_pks"], record)
                if candidate_rank is not None and rerank_rank is not None:
                    rank_gains.append(candidate_rank - rerank_rank)
            rerank_report[dataset] = {
                "rescue_at_5": len(rescues) / len(items),
                "damage_at_5": len(damages) / len(items),
                "mean_best_gold_rank_gain": float(np.mean(rank_gains)) if rank_gains else 0.0,
                "reranker_latency": _percentiles([rerank_by_id[record.query_id]["latency_ms"]["reranker"] for record in items]),
                "rescued_query_ids": rescues,
                "damaged_query_ids": damages,
            }
            route_counts = [len(agent_by_id[record.query_id]["selected_channels"]) for record in items]
            planner_report[dataset] = {
                "avg_routes_per_query": float(np.mean(route_counts)),
                "one_route_ratio": sum(count == 1 for count in route_counts) / len(route_counts),
                "two_route_ratio": sum(count == 2 for count in route_counts) / len(route_counts),
                "three_route_ratio": sum(count == 3 for count in route_counts) / len(route_counts),
                "planner_latency": _percentiles([agent_by_id[record.query_id]["latency_ms"]["planner"] for record in items]),
                "retrieval_latency": _percentiles([agent_by_id[record.query_id]["latency_ms"]["retrieval"] for record in items]),
                "fallback_count": sum(bool(agent_by_id[record.query_id]["planner_error"]) for record in items),
            }
        (output_dir / "reranker_analysis.json").write_text(json.dumps(rerank_report, ensure_ascii=False, indent=2), encoding="utf-8")

        (output_dir / "planner_analysis.json").write_text(json.dumps(planner_report, ensure_ascii=False, indent=2), encoding="utf-8")
        latency = {
            dataset: {
                "fixed_retrieval": _percentiles([fixed_by_id[record.query_id]["latency_ms"]["retrieval"] for record in items]),
                "fixed_rerank": _percentiles([rerank_by_id[record.query_id]["latency_ms"]["reranker"] for record in items]),
                "agent_planner": planner_report[dataset]["planner_latency"],
                "agent_retrieval": planner_report[dataset]["retrieval_latency"],
                "agent_rerank": _percentiles([agent_by_id[record.query_id]["latency_ms"]["reranker"] for record in items]),
            }
            for dataset, items in records_by_dataset.items()
        }
        (output_dir / "latency.json").write_text(json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_failure_cases(records, fixed_by_id, rerank_by_id, agent_by_id, raw_routes, output_dir)
        return metrics

    def _write_failure_cases(self, records: list[EvalRecord], fixed: dict[str, dict], rerank: dict[str, dict],
                             agent: dict[str, dict], raw_rows: list[dict], output_dir: Path) -> None:
        raw_by_id = {item["query_id"]: item for item in raw_rows}
        categories: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            fixed_rank = self._best_rank(record, fixed[record.query_id])
            rerank_rank = self._best_rank(record, rerank[record.query_id])
            agent_rank = self._best_rank(record, agent[record.query_id])
            base = {"query_id": record.query_id, "dataset": record.dataset, "query": record.query,
                    "fixed_rank": fixed_rank, "rerank_rank": rerank_rank, "agent_rank": agent_rank}
            if fixed_rank is None and rerank_rank is not None:
                categories["reranker_rescue"].append(base)
            if fixed_rank is not None and rerank_rank is None:
                categories["reranker_damage"].append(base)
            if rerank_rank is None and agent_rank is not None:
                categories["planner_win"].append(base)
            if rerank_rank is not None and agent_rank is None:
                route_ranks = {name: _rank_of_gold([item["pk"] for item in documents], record)
                               for name, documents in raw_by_id[record.query_id]["routes"].items()}
                categories["planner_loss"].append({**base, "selected_channels": agent[record.query_id]["selected_channels"],
                                                   "gold_rank_summary": route_ranks.get("summary_dense"),
                                                   "gold_rank_text": route_ranks.get("text_dense"),
                                                   "gold_rank_bm25": route_ranks.get("text_sparse")})
            raw = raw_by_id[record.query_id]["routes"]
            bm25_rank = _rank_of_gold([item["pk"] for item in raw["text_sparse"][:5]], record)
            dense_rank = min((rank for rank in (_rank_of_gold([item["pk"] for item in raw[name][:5]], record)
                                                  for name in ("summary_dense", "text_dense")) if rank is not None), default=None)
            if bm25_rank is not None and dense_rank is None:
                categories["bm25_only_strong"].append({**base, "bm25_rank": bm25_rank})
            if dense_rank is not None and bm25_rank is None:
                categories["dense_strong"].append({**base, "dense_rank": dense_rank})
        rows = []
        expected_categories = (
            "reranker_rescue", "reranker_damage", "planner_win", "planner_loss",
            "bm25_only_strong", "dense_strong",
        )
        for category in expected_categories:
            items = categories[category]
            # 即使没有命中案例也保留该类别，artifact 因而能明确区分“无案例”和“未生成”。
            rows.append({"category": category, "case_count": len(items), "cases": items[:5]})
        self._write_jsonl(output_dir / "failure_cases.jsonl", rows)

    def _write_config_snapshot(self, output_dir: Path) -> None:
        snapshot = self.config.model_dump()

        def redact(value: object, key: str = "") -> object:
            if isinstance(value, dict):
                return {item_key: redact(item_value, item_key) for item_key, item_value in value.items()}
            if value is not None and ("token" in key.lower() or "key" in key.lower()):
                return "${" + str(value) + "}"
            return value

        snapshot = redact(snapshot)
        (output_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
