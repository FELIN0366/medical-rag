#!/usr/bin/env python
"""Stage 4 唯一正式 Runner：本地轨迹、真实 Agent 与四项行为评测。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from MedicalRag.agent.MedicalAgent import MedicalAgent
from MedicalRag.agent_evaluation.dataset import load_cases, validate_cases
from MedicalRag.agent_evaluation.runner import AgentEvaluationRunner
from MedicalRag.agent_evaluation.evaluators import aggregate_efficiency
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.core.utils import create_llm_client


def _require_runtime_secrets(config) -> None:
    required = (
        (config.llm.env_key_name, "Agent 与 DeepSeek Judge"),
        (config.embedding.summary_dense.env_key_name, "在线 dense embedding"),
        (config.reranker.api_key_env, "DashScope Reranker"),
        (config.reranker.workspace_id_env, "DashScope Reranker 工作空间"),
    )
    missing = [f"{name}（{purpose}）" for name, purpose in required if name and not os.getenv(name)]
    if missing:
        raise RuntimeError("BLOCKED：缺少真实本地 Eval 所需环境变量：" + "、".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/agent_behavior_cases.jsonl"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/stage4"))
    parser.add_argument("--smoke", type=int, default=0, metavar="N", help="只运行前 N 条真实用例作本地 smoke")
    parser.add_argument("--case-start", type=int, default=0, help="从零开始的用例起始下标，仅用于受限执行环境分片")
    parser.add_argument("--merge-shards", action="store_true", help="只合并 artifact-dir/shards 下的 30 个本地分片，不调用模型")
    parser.add_argument("--case-timeout-seconds", type=float, default=600.0, help="单条 Eval 的硬超时；超时会增量记录失败后继续")
    args = parser.parse_args()
    config = ConfigLoader().config
    all_cases = load_cases(args.dataset)
    validate_cases(all_cases)
    if args.merge_shards:
        _merge_shards(all_cases, args.dataset, args.artifact_dir, config)
        return
    _require_runtime_secrets(config)
    if args.case_start < 0 or args.case_start >= len(all_cases):
        raise ValueError("--case-start 必须位于冻结 Eval Set 范围内")
    cases = all_cases[args.case_start:]
    if args.smoke:
        if args.smoke < 1:
            raise ValueError("--smoke 必须大于 0")
        cases = cases[:args.smoke]
    judge_config = config.llm.model_copy(update={"temperature": 0.0})
    runner = AgentEvaluationRunner(
        config,
        agent_factory=lambda: MedicalAgent(config, power_model=create_llm_client(config.llm)),
        judge_llm=create_llm_client(judge_config),
        case_timeout_seconds=args.case_timeout_seconds,
    )
    metrics = runner.run(cases, args.dataset, args.artifact_dir)
    label = (
        f"本地 smoke（{len(cases)} 条）" if args.smoke
        else (f"正式分片（起始下标 {args.case_start}，{len(cases)} 条）" if args.case_start else "30 条正式本地 Eval")
    )
    print(f"Stage 4 {label}完成。Task Success={metrics['task_success_rate']:.4f}，"
          f"Trajectory={metrics['trajectory_correct_rate']:.4f}")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _merge_shards(cases, dataset: Path, artifact_dir: Path, config) -> None:
    """合并已完成的逐 Case 本地分片，不重新调用 Agent、Embedding、Reranker 或 Judge。"""
    shard_root = artifact_dir / "shards"
    shard_dirs = sorted(path for path in shard_root.glob("case-*") if path.is_dir())
    predictions, results, trajectories = [], [], []
    for directory in shard_dirs:
        predictions.extend(_read_jsonl(directory / "predictions.jsonl"))
        results.extend(_read_jsonl(directory / "evaluator_results.jsonl"))
        trajectories.extend(_read_jsonl(directory / "trajectories.jsonl"))
    by_prediction = {row["case_id"]: row for row in predictions}
    by_result = {row["case_id"]: row for row in results}
    expected_ids = [case.case_id for case in cases]
    if set(by_prediction) != set(expected_ids) or set(by_result) != set(expected_ids):
        raise RuntimeError("无法合并：分片不完整、重复或与冻结 Dataset 不一致")
    predictions = [by_prediction[case_id] for case_id in expected_ids]
    results = [by_result[case_id] for case_id in expected_ids]
    efficiency_rows = [row["efficiency"] for row in results]
    efficiency = aggregate_efficiency(efficiency_rows)
    helper = AgentEvaluationRunner(config, lambda: None, None)
    metrics = helper._metrics(cases, predictions, results, efficiency)
    failures = helper._failure_rows(cases, predictions, results)
    from MedicalRag.agent_evaluation.dataset import freeze_dataset
    freeze_dataset(dataset, artifact_dir)
    _write_jsonl(artifact_dir / "predictions.jsonl", predictions)
    _write_jsonl(artifact_dir / "evaluator_results.jsonl", results)
    _write_jsonl(artifact_dir / "failure_cases.jsonl", failures)
    _write_jsonl(artifact_dir / "trajectories.jsonl", trajectories)
    (artifact_dir / "efficiency.json").write_text(json.dumps(efficiency, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    helper._write_config_snapshot(artifact_dir)
    print("Stage 4 已合并 30 个本地分片，无额外模型调用。")


if __name__ == "__main__":
    main()
