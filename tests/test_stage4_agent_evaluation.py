from __future__ import annotations

import json
from pathlib import Path

from MedicalRag.agent_evaluation.dataset import CATEGORY_COUNTS, load_cases, validate_cases
from MedicalRag.agent_evaluation.evaluators import (
    LLMBehaviorJudge,
    aggregate_efficiency,
    efficiency_record,
    trajectory_tool_correctness,
)
from MedicalRag.agent_evaluation.models import AgentEvalOutput
from MedicalRag.agent_evaluation.runner import AgentEvaluationRunner
from MedicalRag.agent_evaluation.local_trace import LocalTraceCollector
from MedicalRag.config.loader import ConfigLoader


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeJudgeModel:
    def invoke(self, messages):
        prompt = messages[0].content
        if "task_success" in prompt:
            return FakeResponse('{"task_success": 1, "reason": "符合预期"}')
        return FakeResponse('{"planning_quality": 2, "reason": "覆盖完整"}')


def _cases():
    return load_cases(Path("data/eval/agent_behavior_cases.jsonl"))


def test_frozen_dataset_has_exact_category_distribution():
    cases = _cases()
    report = validate_cases(cases)
    assert report["case_count"] == 30
    assert report["category_counts"] == CATEGORY_COUNTS
    assert len({case.case_id for case in cases}) == 30


def test_structured_judges_and_trajectory_rules():
    case = _cases()[0]
    output = AgentEvalOutput(case_id=case.case_id, final_answer="高血压可有头晕。", subqueries=["高血压症状"])
    judge = LLMBehaviorJudge(FakeJudgeModel())
    assert judge.task_success(case, output).task_success == 1
    assert judge.planning_quality(case, output).planning_quality == 2
    result = trajectory_tool_correctness(case, output)
    assert result["trajectory_correct"] == 1
    assert result["decompose_rule_pass"] is True


def test_clarification_planning_is_na_and_excluded_from_mean():
    case = next(item for item in _cases() if item.primary_category == "clarification_needed")
    output = AgentEvalOutput(case_id=case.case_id, needs_clarification=True, clarification_questions=["请问血压数值？"])
    judge = LLMBehaviorJudge(FakeJudgeModel())
    assert judge.planning_quality(case, output) is None
    summary = aggregate_efficiency([efficiency_record(case, output)])
    assert summary["token_usage"] == {"available_count": 0, "mean": None}


def test_runner_reuses_a_session_for_multi_turn_case():
    case = next(item for item in _cases() if item.primary_category == "multi_turn_context")
    calls = []

    class FakeAgent:
        def set_local_trace_collector(self, _collector, **kwargs):
            self.context = kwargs

        def answer(self, turn):
            calls.append((self.context["session_id"], turn))
            return {
                "ask_obj": None,
                "final_answer": "回答",
                "rewritten_query": "改写",
                "sub_query": None,
                "sub_query_results": [{"query": "子查询", "retrieval_info": {"selected_channels": ["text_sparse"], "db_search_count": 1}, "web_search_count": 0, "judge_retry_count": 0}],
            }

    runner = AgentEvaluationRunner(ConfigLoader().config, FakeAgent, FakeJudgeModel())
    runner.run_case(case, LocalTraceCollector())
    assert len(calls) == len(case.turns)
    assert len({session for session, _ in calls}) == 1


def test_local_trace_does_not_persist_retrieved_documents(tmp_path):
    collector = LocalTraceCollector()
    collector.emit({"case_id": "case", "session_id": "session", "turn_index": 1}, "retrieval_executor",
                   candidate_count=30, documents=["不应写入的检索内容"])
    path = tmp_path / "trajectories.jsonl"
    collector.write_jsonl(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["event"] == "retrieval_executor"
    assert "documents" not in row["data"]


def test_runner_writes_complete_local_artifacts(tmp_path, monkeypatch):
    cases = _cases()
    config = ConfigLoader().config
    runner = AgentEvaluationRunner(config, lambda: None, FakeJudgeModel())

    def fake_run_case(case, _collector):
        return AgentEvalOutput(case_id=case.case_id, final_answer="回答", subqueries=["查询"], db_search_count=1)

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    metrics = runner.run(cases, Path("data/eval/agent_behavior_cases.jsonl"), tmp_path)
    expected = {
        "agent_eval_dataset.jsonl", "dataset_report.json", "predictions.jsonl", "evaluator_results.jsonl",
        "metrics.json", "efficiency.json", "failure_cases.jsonl", "trajectories.jsonl", "config_snapshot.yaml",
        "progress.json",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert metrics["case_count"] == 30
    assert len((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 30
