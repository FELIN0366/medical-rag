#!/usr/bin/env python
"""校验并冻结 Stage 4 本地 Agent Behavior Eval Set，不调用任何远程服务。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from MedicalRag.agent_evaluation.dataset import freeze_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/agent_behavior_cases.jsonl"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/stage4"))
    args = parser.parse_args()
    report = freeze_dataset(args.dataset, args.artifact_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
