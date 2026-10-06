"""Stage 4 唯一 Agent Eval Runner 的本地执行与 artifact 写入逻辑。"""
from __future__ import annotations

import json
import time
import signal
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable

import yaml
from langchain_core.language_models.chat_models import BaseChatModel

from ..agent.MedicalAgent import MedicalAgent
from ..config.models import AppConfig
from .dataset import freeze_dataset
from .evaluators import LLMBehaviorJudge, aggregate_efficiency, efficiency_record, trajectory_tool_correctness
from .local_trace import LocalTraceCollector
from .models import AgentEvalCase, AgentEvalOutput


class AgentEvaluationRunner:
    """顺序执行每个 Case；同一多轮 Case 固定复用一个 Agent 实例与本地 session。"""

    def __init__(self, config: AppConfig, agent_factory: Callable[[], MedicalAgent], judge_llm: BaseChatModel,
                 case_timeout_seconds: float = 600.0) -> None:
        self.config = config
        self.agent_factory = agent_factory
        self.judge = LLMBehaviorJudge(judge_llm)
        self.case_timeout_seconds = case_timeout_seconds

    def _run_case_with_timeout(self, case: AgentEvalCase, collector: LocalTraceCollector) -> AgentEvalOutput:
        """仅 Eval Runner 使用的 Case 总时限；超时仍保留已增量落盘的 span。"""
        if self.case_timeout_seconds <= 0:
            return self.run_case(case, collector)
        def raise_timeout(_signum, _frame):
            raise TimeoutError(f"Case 超过 {self.case_timeout_seconds} 秒")
        previous = signal.signal(signal.SIGALRM, raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.case_timeout_seconds)
        try:
            return self.run_case(case, collector)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def run_case(self, case: AgentEvalCase, collector: LocalTraceCollector) -> AgentEvalOutput:
        session_id = f"stage4-{case.case_id}"
        context = {
            "case_id": case.case_id, "primary_category": case.primary_category,
            "session_id": session_id, "turn_index": 0,
        }
        with collector.span(context, "agent_initialization", operation="create_agent"):
            agent = self.agent_factory()
        agent.set_local_trace_collector(
            collector,
            case_id=case.case_id,
            primary_category=case.primary_category,
            session_id=session_id,
        )
        started = time.perf_counter()
        db_search_count = web_search_count = retry_count = 0
        final_state: dict = {}
        for turn in case.turns:
            state = agent.answer(turn)
            final_state = state
            sub_results = state.get("sub_query_results", [])
            db_search_count += sum(int(result.get("retrieval_info", {}).get("db_search_count", 0)) for result in sub_results)
            web_search_count += sum(int(result.get("web_search_count", 0)) for result in sub_results)
            retry_count += sum(int(result.get("judge_retry_count", 0)) for result in sub_results)
            ask_obj = state.get("ask_obj")
            # 原始 MedicalAgent 的追问只结束当前 invocation；多轮 Eval 必须把后续
            # 冻结用户输入继续送入同一个 Agent 实例，保留其会话状态。
            if (
                case.primary_category == "clarification_needed"
                and case.expectation.terminal_behavior == "clarification"
                and ask_obj
                and ask_obj.need_ask
            ):
                break
        ask_obj = final_state.get("ask_obj")
        sub_results = final_state.get("sub_query_results", [])
        split = final_state.get("sub_query")
        subqueries = [result.get("query", "") for result in sub_results if result.get("query")]
        if not subqueries and split and split.need_split:
            subqueries = list(split.sub_query)
        if not subqueries and split and split.rewrite_query:
            subqueries = [split.rewrite_query]
        return AgentEvalOutput(
            case_id=case.case_id,
            final_answer=final_state.get("final_answer", ""),
            needs_clarification=bool(ask_obj and ask_obj.need_ask),
            clarification_questions=list(ask_obj.questions) if ask_obj and ask_obj.need_ask else [],
            rewritten_query=final_state.get("rewritten_query", "") or (split.rewrite_query if split else ""),
            subqueries=subqueries,
            selected_channels=[list(result.get("retrieval_info", {}).get("selected_channels", [])) for result in sub_results],
            db_search_count=db_search_count,
            web_search_count=web_search_count,
            retry_count=retry_count,
            latency_ms=(time.perf_counter() - started) * 1000,
            trajectory_id=session_id,
            tool_call_count=db_search_count + web_search_count,
        )

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(self, cases: list[AgentEvalCase], source_dataset: Path, artifact_dir: Path) -> dict:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        freeze_dataset(source_dataset, artifact_dir)
        collector = LocalTraceCollector(artifact_dir / "trajectories.jsonl")
        predictions: list[dict] = []
        evaluator_rows: list[dict] = []
        efficiency_rows: list[dict] = []
        for index, case in enumerate(cases, start=1):
            print(f"Stage 4 正在执行 {index}/{len(cases)}：{case.case_id}", flush=True)
            case_started = time.perf_counter()
            try:
                output = self._run_case_with_timeout(case, collector)
            except Exception as error:
                output = AgentEvalOutput(case_id=case.case_id, final_answer="",
                                        latency_ms=(time.perf_counter() - case_started) * 1000)
                prediction = {"primary_category": case.primary_category, "turns": case.turns,
                              "execution_error": type(error).__name__, **output.model_dump()}
                predictions.append(prediction)
                trajectory = trajectory_tool_correctness(case, output)
                evaluator_rows.append({
                    "case_id": case.case_id, "task_success": 0, "task_success_reason": "技术失败，未产生 Agent 输出。",
                    "planning_quality": None, "planning_quality_reason": "技术失败。", **trajectory,
                    "efficiency": efficiency_record(case, output), "execution_error": type(error).__name__,
                })
                efficiency_rows.append(efficiency_record(case, output))
                collector.emit({"case_id": case.case_id, "primary_category": case.primary_category,
                                "session_id": f"stage4-{case.case_id}"}, "case_error",
                               error_type=type(error).__name__, error_message=str(error)[:300])
                self._write_jsonl(artifact_dir / "predictions.jsonl", predictions)
                self._write_jsonl(artifact_dir / "evaluator_results.jsonl", evaluator_rows)
                (artifact_dir / "progress.json").write_text(json.dumps({
                    "completed_cases": len(predictions), "last_case_id": case.case_id,
                    "last_case_error": type(error).__name__,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            prediction = {"primary_category": case.primary_category, "turns": case.turns, **output.model_dump()}
            predictions.append(prediction)
            task = self.judge.task_success(case, output)
            planning = self.judge.planning_quality(case, output)
            trajectory = trajectory_tool_correctness(case, output)
            efficiency = efficiency_record(case, output)
            evaluator_rows.append({
                "case_id": case.case_id,
                "task_success": task.task_success,
                "task_success_reason": task.reason,
                "planning_quality": planning.planning_quality if planning else None,
                "planning_quality_reason": planning.reason if planning else "N/A：追问用例未进入规划。",
                **trajectory,
                "efficiency": efficiency,
            })
            efficiency_rows.append(efficiency)
            # 每个 Case 结束后立即覆盖写入已完成结果；中断也保留可审计进度。
            self._write_jsonl(artifact_dir / "predictions.jsonl", predictions)
            self._write_jsonl(artifact_dir / "evaluator_results.jsonl", evaluator_rows)
            (artifact_dir / "progress.json").write_text(json.dumps({
                "completed_cases": len(predictions), "last_case_id": case.case_id,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_jsonl(artifact_dir / "predictions.jsonl", predictions)
        self._write_jsonl(artifact_dir / "evaluator_results.jsonl", evaluator_rows)
        self._write_jsonl(artifact_dir / "failure_cases.jsonl", self._failure_rows(cases, predictions, evaluator_rows))
        # Collector 已增量写入；此处重写同一完整文件以支持内存模式和测试。
        collector.write_jsonl(artifact_dir / "trajectories.jsonl")
        efficiency = aggregate_efficiency(efficiency_rows)
        (artifact_dir / "efficiency.json").write_text(json.dumps(efficiency, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics = self._metrics(cases, predictions, evaluator_rows, efficiency)
        (artifact_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_config_snapshot(artifact_dir)
        return metrics

    def _metrics(self, cases: list[AgentEvalCase], predictions: list[dict], rows: list[dict], efficiency: dict) -> dict:
        by_category: dict[str, list[dict]] = defaultdict(list)
        for case, row in zip(cases, rows):
            by_category[case.primary_category].append(row)

        def summarize(items: list[dict]) -> dict:
            planning = [item["planning_quality"] for item in items if item["planning_quality"] is not None]
            return {
                "case_count": len(items),
                "task_success_rate": mean(item["task_success"] for item in items),
                "planning_quality": {
                    "applicable_count": len(planning),
                    "mean": mean(planning) if planning else None,
                    "good_ratio": sum(item == 2 for item in planning) / len(planning) if planning else None,
                },
                "trajectory_correct_rate": mean(item["trajectory_correct"] for item in items),
            }

        overall = summarize(rows)
        return {
            "case_count": len(cases),
            **overall,
            **efficiency,
            "by_primary_category": {name: summarize(items) for name, items in by_category.items()},
        }

    @staticmethod
    def _failure_rows(cases: list[AgentEvalCase], predictions: list[dict], rows: list[dict]) -> list[dict]:
        categories: dict[str, list[dict]] = defaultdict(list)
        for case, prediction, result in zip(cases, predictions, rows):
            actual = {
                "subqueries": prediction["subqueries"],
                "db_search_count": prediction["db_search_count"],
                "web_search_count": prediction["web_search_count"],
                "retry_count": prediction["retry_count"],
            }
            base = {"case_id": case.case_id, "turns": case.turns, "expected_behavior": case.expectation.model_dump(),
                    "actual_planning": actual, "final_answer": prediction["final_answer"],
                    "trajectory_id": prediction["trajectory_id"]}
            if not result["task_success"]:
                categories["task_failure"].append(base)
            if not result["decompose_rule_pass"]:
                key = "missing_decomposition" if case.expectation.need_decompose else "unnecessary_decomposition"
                categories[key].append(base)
            if not result["web_rule_pass"]:
                key = "missing_web" if case.expectation.web_policy == "required" else "unnecessary_web"
                categories[key].append(base)
            if not result["retry_rule_pass"]:
                categories["retry_overuse"].append(base)
            if result.get("planning_quality") == 0:
                categories["planning_error"].append(base)
        expected_categories = (
            "planning_error", "unnecessary_decomposition", "missing_decomposition", "unnecessary_web",
            "missing_web", "retry_overuse", "task_failure",
        )
        return [
            {"category": name, "case_count": len(categories[name]), "cases": categories[name][:5]}
            for name in expected_categories
        ]

    def _write_config_snapshot(self, artifact_dir: Path) -> None:
        snapshot = self.config.model_dump()
        def redact(value, key=""):
            if isinstance(value, dict):
                return {name: redact(item, name) for name, item in value.items()}
            if value is not None and ("token" in key.lower() or "key" in key.lower()):
                return "${" + str(value) + "}"
            return value
        (artifact_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(redact(snapshot), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
