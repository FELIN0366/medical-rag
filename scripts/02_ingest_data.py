#!/usr/bin/env python
"""构建 Stage 1 检索单元并写入当前 Milvus Schema。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase
from MedicalRag.ingestion.pipeline import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--qa-sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["structural", "fixed", "recursive"], default="structural")
    parser.add_argument("--recreate", action="store_true", help="删除并重建 Stage 1 Collection")
    args = parser.parse_args()
    started = time.perf_counter()
    corpus = build_corpus(args.data_dir, qa_sample_size=args.qa_sample_size, seed=args.seed, strategy=args.strategy)
    if corpus.parse_failures:
        raise RuntimeError(f"Refusing to ingest after parse failures: {corpus.parse_failures}")
    config = ConfigLoader().config
    kb = MedicalHybridKnowledgeBase(config)
    kb._create_collection(recreate=args.recreate)
    inserted = kb.add_chunks(corpus.chunks)
    # Docker Milvus 的写入可见性是异步的；先 flush 再建索引和读取行数，
    # 避免刚完成 upsert 就将暂时的 row_count=0 误判为入库失败。
    kb.client.flush(config.milvus.collection_name)
    kb.build_index()
    deadline = time.monotonic() + 20
    stats = kb.client.get_collection_stats(config.milvus.collection_name)
    while int(stats.get("row_count", 0)) <= 0 and time.monotonic() < deadline:
        time.sleep(0.25)
        stats = kb.client.get_collection_stats(config.milvus.collection_name)
    report = {
        "collection_name": config.milvus.collection_name, "recreated": args.recreate,
        "qa_rows": len(corpus.qa_chunks), "literature_chunks": len(corpus.literature_chunks),
        "total_rows_inserted": inserted, "collection_row_count": int(stats.get("row_count", 0)),
        "summary_dense_dim": config.embedding.summary_dense.dimension,
        "text_dense_dim": config.embedding.text_dense.dimension,
        "sparse_bm25": "Milvus built-in BM25", "strategy": args.strategy,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if not kb.client.has_collection(config.milvus.collection_name) or report["collection_row_count"] <= 0:
        raise RuntimeError(f"Milvus collection verification failed: {report}")
    output = Path("artifacts/stage1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "ingestion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
