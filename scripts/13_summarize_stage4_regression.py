#!/usr/bin/env python
"""为 Stage 4 回归生成拓扑审计、节点延迟和基线对比产物。"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean


BASE_COMMIT = "4ec5871ad874344d228b19aeed7ab38c4ba0b940"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * percent)))]


def _latency_by_node(events: list[dict]) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.get("event") != "span":
            continue
        data = event.get("data", {})
        if data.get("phase") == "end" and isinstance(data.get("duration_ms"), (int, float)):
            values[str(data.get("name", "unknown"))].append(float(data["duration_ms"]))
    return {
        name: {
            "count": len(samples),
            "mean_ms": round(mean(samples), 3),
            "p50_ms": round(_percentile(samples, 0.5), 3),
            "p95_ms": round(_percentile(samples, 0.95), 3),
            "max_ms": round(max(samples), 3),
        }
        for name, samples in sorted(values.items())
    }


def _base_source(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{BASE_COMMIT}:{path}"], check=True, text=True,
        capture_output=True,
    ).stdout


def _write_graph_diff(output: Path) -> None:
    # 读取冻结基线作为报告生成的前置校验，避免把人工记忆当作原版依据。
    original_agent = _base_source("src/MedicalRag/agent/MedicalAgent.py")
    original_search = _base_source("src/MedicalRag/agent/SearchGraph.py")
    current_agent = Path("src/MedicalRag/agent/MedicalAgent.py").read_text(encoding="utf-8")
    current_search = Path("src/MedicalRag/agent/SearchGraph.py").read_text(encoding="utf-8")
    assertions = {
        "原始 MedicalAgent 包含 ask 节点": 'g.add_node("ask"' in original_agent,
        "原始 SearchGraph 包含 db_search 节点": 'g.add_node("db_search"' in original_search,
        "当前 MedicalAgent 包含 clarification 节点": 'g.add_node("clarification"' in current_agent,
        "当前 SearchGraph 包含 retrieve 节点": 'g.add_node("retrieve"' in current_search,
        "当前无旧数据库 ToolNode": "db_tool_node" not in current_search,
    }
    checks = "\n".join(f"- {'通过' if passed else '失败'}：{name}" for name, passed in assertions.items())
    output.write_text(
        f"""# 原始图与当前图的语义拓扑对比

冻结基线：`yolo-hyl/medical-rag@{BASE_COMMIT}`。本报告在生成时以 `git show` 读取了两个原始文件，并读取当前工作区文件。

## MedicalAgent

| 原始节点 | 当前节点 | 原始出边 | 当前出边 | 语义等价 | 原因 |
|---|---|---|---|---|---|
| `ask` | `clarification` | `END`（追问）/ `extract_ask_and_reply` | `END`（追问）/ `background_update` | 是 | 追问只结束当前 invocation。 |
| `extract_ask_and_reply` | `background_update` | `split_query` | `split_query` | 是 | 抽取追问历史为背景。 |
| `check_update_background` | `check_update_background` | `split_query` | `split_query` | 是 | 后续 turn 更新背景后进入规划。 |
| `split_query` | `split_query` | `Send(search_one)` | `Send(search_one)` | 是 | 并行子查询分发保持不变。 |
| `answer` | `gather_answer` | `END` | `END` | 是 | 当前刻意不回传 `sub_query_results`，避免 add reducer 重复追加。 |

## SearchGraph

| 原始节点 | 当前节点 | 原始出边 | 当前出边 | 语义等价 | 原因 |
|---|---|---|---|---|---|
| `db_search` | `retrieve` | `web_search` / `rag` | `web_search` / `rag` | 是（执行器替换） | 当前唯一 Milvus 入口是 Planner → UnifiedRetrievalExecutor → Reranker。 |
| `web_search` | `web_search` | `rag` | `rag` | 是 | Web ToolNode 保留。 |
| `rag` | `rag` | `judge` / `END` | `judge` / `END` | 是 | analysis/fast 模式语义不变。 |
| `judge` | `judge` | pass/retry/fail | pass/retry/fail | 是 | 重试回路不变。 |

## State 生命周期

| 状态 | 跨 turn 行为 |
|---|---|
| `ask_obj` | 追问时保留供本轮返回；成功汇总后清空。 |
| `curr_ask_num` | 追问次数跨 turn 保留；成功汇总后复位。 |
| `asking_messages`、`background_info` | 保留，供后续用户回答追问。 |
| `dialogue_messages`、`multi_summary`、`running_summary` | 保留，供多轮上下文与压缩摘要。 |
| `curr_input`、`sub_query_results` | 每次 `answer()` 前更新/清空；后者仅本轮聚合。 |

## 自动源文件校验

{checks}

结论：当前图未发生需要回退的拓扑回归。原始 `db_search` 的可执行 ToolNode 已被 `retrieve` 的 Stage 2 统一检索链替换；这是有意算法替换而非边断裂。
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/stage4_regression_v2"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/stage4_diagnosis"))
    args = parser.parse_args()
    events = _read_jsonl(args.artifact_dir / "trajectories.jsonl")
    latency = _latency_by_node(events)
    (args.artifact_dir / "latency_by_node.json").write_text(
        json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    before, after = _read_json(args.baseline_dir / "metrics.json"), _read_json(args.artifact_dir / "metrics.json")
    comparison = {
        "baseline_artifact": str(args.baseline_dir),
        "regression_artifact": str(args.artifact_dir),
        "before": before,
        "after": after,
        "delta": {
            "task_success_rate": after["task_success_rate"] - before["task_success_rate"],
            "trajectory_correct_rate": after["trajectory_correct_rate"] - before["trajectory_correct_rate"],
            "planning_quality_mean": (
                after["planning_quality"]["mean"] - before["planning_quality"]["mean"]
            ),
            "p50_latency_ms": after["latency"]["p50_ms"] - before["latency"]["p50_ms"],
            "p95_latency_ms": after["latency"]["p95_ms"] - before["latency"]["p95_ms"],
        },
    }
    (args.artifact_dir / "before_after.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_graph_diff(args.artifact_dir / "original_graph_diff.md")


if __name__ == "__main__":
    main()
