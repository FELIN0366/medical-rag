"""冻结 Stage 4 行为数据集的读取、校验与复制。"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .models import AgentEvalCase

CATEGORY_COUNTS = {
    "simple_single_hop": 5,
    "rewrite_needed": 5,
    "decompose_multi_intent": 5,
    "clarification_needed": 4,
    "kb_sufficient_no_web": 4,
    "kb_insufficient_web": 3,
    "multi_turn_context": 4,
}


def load_cases(path: Path) -> list[AgentEvalCase]:
    with path.open(encoding="utf-8") as stream:
        return [AgentEvalCase.model_validate_json(line) for line in stream if line.strip()]


def validate_cases(cases: list[AgentEvalCase]) -> dict:
    if len(cases) != 30:
        raise ValueError(f"Stage 4 Eval Set 必须恰好 30 条，实际为 {len(cases)}")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Stage 4 Eval Set 的 case_id 必须唯一")
    actual = Counter(case.primary_category for case in cases)
    if dict(actual) != CATEGORY_COUNTS:
        raise ValueError(f"类别数量不符合冻结分布：实际={dict(actual)}，期望={CATEGORY_COUNTS}")
    for case in cases:
        expectation = case.expectation
        if expectation.terminal_behavior == "clarification" and len(case.turns) != 1:
            raise ValueError(f"追问用例必须在首轮结束：{case.case_id}")
        if expectation.need_decompose and not expectation.expected_topics:
            raise ValueError(f"拆解用例必须给出 expected_topics：{case.case_id}")
        if expectation.web_policy == "required" and "web_search" in expectation.forbidden_tools:
            raise ValueError(f"Web required 与 forbidden_tools 冲突：{case.case_id}")
    return {"case_count": len(cases), "category_counts": dict(actual)}


def freeze_dataset(source: Path, output_dir: Path) -> dict:
    cases = load_cases(source)
    report = validate_cases(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    target = output_dir / "agent_eval_dataset.jsonl"
    target.write_bytes(content)
    report.update({
        "source": str(source),
        "frozen_copy": str(target),
        "sha256": hashlib.sha256(content).hexdigest(),
    })
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
