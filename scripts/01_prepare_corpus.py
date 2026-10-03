#!/usr/bin/env python
"""构建可复现的 Stage 1 语料，不执行 embedding 或 Milvus 写入。"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from MedicalRag.ingestion.pipeline import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--qa-sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["structural", "fixed", "recursive"], default="structural")
    args = parser.parse_args()
    started = __import__("time").perf_counter()
    corpus = build_corpus(args.data_dir, qa_sample_size=args.qa_sample_size, seed=args.seed, strategy=args.strategy)
    output = Path("artifacts/stage1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "corpus_inventory.json").write_text(json.dumps(corpus.inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    lengths = [len(chunk.document) for chunk in corpus.literature_chunks]
    kinds = {kind: sum(1 for item in corpus.inventory if item["kind"] == kind and item["status"] == "supported")
             for kind in ("pdf", "html", "markdown")}
    report = {
        "qa_sampled_count": len(corpus.qa_chunks), "seed": args.seed, "strategy": args.strategy,
        "pdf_count": kinds["pdf"], "html_count": kinds["html"], "markdown_count": kinds["markdown"],
        "ignored_unsupported_file_count": sum(1 for item in corpus.inventory if item["status"] == "ignored_unsupported"),
        "ignored_non_corpus_file_count": sum(1 for item in corpus.inventory if item["status"] == "ignored_non_corpus"),
        "parse_failure_count": len(corpus.parse_failures), "parse_failures": corpus.parse_failures,
        "parsed_document_count": len(corpus.documents), "literature_chunk_count": len(corpus.literature_chunks),
        "total_unit_count": len(corpus.chunks), "mean_chunk_chars": round(statistics.mean(lengths), 2) if lengths else 0,
        "p50_chunk_chars": statistics.median(lengths) if lengths else 0,
        "p95_chunk_chars": sorted(lengths)[max(0, int(len(lengths) * .95) - 1)] if lengths else 0,
        "table_chunk_count": sum(1 for chunk in corpus.literature_chunks if "表格" in chunk.section_path),
        "warnings": {doc.source_name: doc.warnings for doc in corpus.documents if doc.warnings},
        "elapsed_seconds": round(__import__("time").perf_counter() - started, 3),
    }
    (output / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    samples = corpus.qa_chunks[:3] + corpus.literature_chunks[:3]
    with (output / "chunk_samples.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in samples:
            stream.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n样本：")
    for chunk in samples:
        print(json.dumps({"source": chunk.source, "source_name": chunk.source_name, "title": chunk.title,
                          "section_path": chunk.section_path, "summary": chunk.summary,
                          "document_preview": chunk.document[:180]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
