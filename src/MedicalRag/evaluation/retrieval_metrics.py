"""Stage 2 检索指标：direct Gold 的 Recall/MRR 与 2/1/0 graded nDCG。"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .retrieval_dataset import EvalRecord


def _pks(prediction: dict, limit: int | None = None) -> list[str]:
    values = prediction.get("final_pks", prediction.get("pks", []))
    return list(values[:limit]) if limit is not None else list(values)


def candidate_pks(prediction: dict, limit: int = 30) -> list[str]:
    """候选召回始终基于重排前的 Top-30，而非最终 Top-5。"""
    return list(prediction.get("candidate_pks", _pks(prediction))[:limit])


def recall_at_k(record: EvalRecord, prediction: dict, k: int) -> float:
    return float(bool(record.direct_gold_pks() & set(_pks(prediction, k))))


def reciprocal_rank(record: EvalRecord, prediction: dict, k: int) -> float:
    gold = record.direct_gold_pks()
    for rank, pk in enumerate(_pks(prediction, k), start=1):
        if pk in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(record: EvalRecord, prediction: dict, k: int) -> float:
    gains = [record.relevance(pk) for pk in _pks(prediction, k)]
    dcg = sum((2 ** rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(gains, start=1))
    ideal = sorted([2] * len(record.direct_gold_pks()) + [1] * len(record.partial_relevant_pks), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate_predictions(records: Iterable[EvalRecord], predictions: Iterable[dict]) -> dict:
    """按 QA/Literature 分组输出规范冻结的主指标。"""
    by_id = {item["query_id"]: item for item in predictions}
    groups: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in records:
        groups[record.dataset].append(record)
    report = {}
    for dataset, items in groups.items():
        missing = [record.query_id for record in items if record.query_id not in by_id]
        if missing:
            raise RuntimeError(f"缺少预测结果：{missing}")
        report[dataset] = {
            "query_count": len(items),
            "candidate_recall_at_30": sum(
                float(bool(record.direct_gold_pks() & set(candidate_pks(by_id[record.query_id], 30))))
                for record in items
            ) / len(items),
            "recall_at_5": sum(recall_at_k(record, by_id[record.query_id], 5) for record in items) / len(items),
            "mrr": sum(reciprocal_rank(record, by_id[record.query_id], 5) for record in items) / len(items),
            "ndcg_at_5": sum(ndcg_at_k(record, by_id[record.query_id], 5) for record in items) / len(items),
        }
    return report
