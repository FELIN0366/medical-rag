#!/usr/bin/env python
"""从冻结的 Stage 1 collection 构建 Stage 2 Eval Set；本脚本不写 Milvus。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymilvus import MilvusClient

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.evaluation.retrieval_dataset import (
    EvalRecord,
    build_best_effort_cross_records,
    build_literature_records,
    build_qa_records,
    load_eval_records,
    write_eval_records,
)


def _read_rows(client: MilvusClient, collection: str, source: str, fields: list[str]) -> list[dict]:
    """只读取 collection 中已有行，并显式限制在当前冻结的两个 source。"""
    return client.query(collection_name=collection, filter=f'source == "{source}"', output_fields=fields, limit=1000)


def _validate_cross_records(path: Path, literature_rows: dict[str, dict], qa_rows: list[dict]) -> list[EvalRecord]:
    records = load_eval_records(path)
    if not 5 <= len(records) <= 10:
        raise RuntimeError(f"Cross-corpus diagnostic 必须有 5~10 条人工确认记录，实际为 {len(records)}")
    qa_questions = {str(row["summary"]) for row in qa_rows}
    original_questions = [record.original_question for record in records]
    if len(set(original_questions)) != len(records):
        raise RuntimeError("Cross-corpus 不能用同一 QA 重复凑足样本数")
    for record in records:
        if record.dataset != "cross_corpus" or not record.original_question:
            raise RuntimeError(f"Cross-corpus 记录格式无效：{record.query_id}")
        if record.original_question not in qa_questions:
            raise RuntimeError(f"Cross-corpus 原题不属于冻结 QA：{record.query_id}")
        if record.label_quality == "direct" and not record.direct_gold_pks():
            raise RuntimeError(f"Cross-corpus 缺少 direct Gold：{record.query_id}")
        if record.label_quality == "best_effort" and (record.direct_gold_pks() or not record.partial_relevant_pks):
            raise RuntimeError(f"最佳努力 Cross 必须只保留 partial 标签：{record.query_id}")
        if any(pk not in literature_rows for pk in record.direct_gold_pks() | set(record.partial_relevant_pks)):
            raise RuntimeError(f"Cross-corpus Gold 必须是文献块：{record.query_id}")
    return records


def _write_cross_candidates(client: MilvusClient, collection: str, qa_rows: list[dict], output_path: Path) -> None:
    """以 BM25 只读检索产生人工筛选候选，不把候选误当作 Gold。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for qa in sorted(qa_rows, key=lambda row: row["pk"]):
            hits = client.search(
                collection_name=collection,
                data=[qa["summary"]],
                anns_field="text_sparse",
                filter='source == "literature"',
                limit=3,
                output_fields=["pk", "source_name", "section_path", "document"],
                search_params={"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}},
            )[0]
            for rank, hit in enumerate(hits, start=1):
                entity = hit.get("entity", hit)
                stream.write(json.dumps({
                    "qa_pk": qa["pk"],
                    "original_question": qa["summary"],
                    "candidate_rank_bm25": rank,
                    "candidate_pk": entity.get("pk", hit.get("id", "")),
                    "source_name": entity.get("source_name", ""),
                    "section_path": entity.get("section_path", ""),
                    "evidence_text": entity.get("document", ""),
                }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-corpus-labels", type=Path,
                        help="人工核验的 5~10 条 Cross-corpus JSONL；Gold 必须为直接回答 QA 的文献块")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage2"))
    parser.add_argument("--cross-corpus-candidates-only", action="store_true",
                        help="仅生成供人工核验的 BM25 候选，不生成正式 Eval Set")
    parser.add_argument("--allow-pending-cross-corpus", action="store_true",
                        help="仅在缺少人工 Cross 标签时写出真实 QA/Literature Eval，不生成伪造的 Cross Eval")
    parser.add_argument("--build-best-effort-cross", action="store_true",
                        help="生成只含 partial 标签的最佳努力 Cross 诊断集；不作为 direct-evidence 主评测")
    args = parser.parse_args()

    config = ConfigLoader().config
    client = MilvusClient(uri=config.milvus.uri, token=config.milvus.token)
    if not client.has_collection(config.milvus.collection_name):
        raise RuntimeError("未发现冻结的 Stage 1 collection；禁止在本脚本中创建或重建 collection")
    qa_rows = _read_rows(client, config.milvus.collection_name, "qa", ["pk", "summary"])
    literature = _read_rows(client, config.milvus.collection_name, "literature",
                            ["pk", "source_name", "section_path", "document"])
    literature_by_pk = {row["pk"]: row for row in literature}
    _write_cross_candidates(client, config.milvus.collection_name, qa_rows,
                            args.output_dir / "cross_corpus_candidates.jsonl")
    if args.build_best_effort_cross:
        cross_records = build_best_effort_cross_records(literature_by_pk, qa_rows)
        write_eval_records(args.output_dir / "eval_cross_corpus.jsonl", cross_records)
        print(f"已生成最佳努力 Cross-corpus 诊断集：{len(cross_records)} 条，仅含 partial 标签。")
        return
    if args.cross_corpus_candidates_only:
        print(f"已生成 Cross-corpus 人工核验候选：{args.output_dir / 'cross_corpus_candidates.jsonl'}")
        return
    if args.cross_corpus_labels is None and not args.allow_pending_cross_corpus:
        raise RuntimeError("请先人工审核 cross_corpus_candidates.jsonl，并通过 --cross-corpus-labels 提供 5~10 条 direct-evidence 标签；如仅需生成 QA/Literature Eval，请显式传入 --allow-pending-cross-corpus")
    qa_records = build_qa_records(qa_rows, config)
    literature_records = build_literature_records(literature_by_pk)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # QA 和 Literature 均为真实生成结果；它们可在 Cross 标签尚未完成时独立复核。
    write_eval_records(output_dir / "eval_qa.jsonl", qa_records)
    write_eval_records(output_dir / "eval_literature.jsonl", literature_records)
    if args.cross_corpus_labels is None:
        print(f"已生成 Eval Set：QA={len(qa_records)}，Literature={len(literature_records)}；Cross-corpus 待人工审核")
        return
    cross_records = _validate_cross_records(args.cross_corpus_labels, literature_by_pk, qa_rows)
    # 保留用户人工标签的完整字段，同时把冻结副本放入规范 artifact 位置。
    write_eval_records(output_dir / "eval_cross_corpus.jsonl", cross_records)
    print(f"已生成 Eval Set：QA={len(qa_records)}，Literature={len(literature_records)}，Cross={len(cross_records)}")


if __name__ == "__main__":
    main()
