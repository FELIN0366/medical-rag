#!/usr/bin/env python
"""运行 Stage 2 的 Exp A/B/C；只读取既有 Milvus collection。"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.evaluation.retrieval_dataset import load_eval_records
from MedicalRag.evaluation.retrieval_runner import RetrievalEvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/stage2"))
    args = parser.parse_args()
    paths = [args.artifact_dir / name for name in ("eval_qa.jsonl", "eval_literature.jsonl", "eval_cross_corpus.jsonl")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("缺少冻结 Eval Set，请先运行 09_prepare_stage2_eval.py：" + ", ".join(missing))
    records = [record for path in paths for record in load_eval_records(path)]
    config = ConfigLoader().config
    runner = RetrievalEvaluationRunner(config)
    report = asyncio.run(runner.run(records, args.artifact_dir))
    print("Stage 2 检索评测完成。")
    for name, result in report.items():
        print(name, result)


if __name__ == "__main__":
    main()
