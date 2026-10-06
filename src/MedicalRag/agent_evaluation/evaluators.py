"""Stage 4 的四个行为评测器；Judge 仅接收冻结用例与 Agent 输出。"""
from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .models import AgentEvalCase, AgentEvalOutput


class TaskJudgeResult(BaseModel):
    task_success: int = Field(ge=0, le=1)
    reason: str


class PlanningJudgeResult(BaseModel):
    planning_quality: int = Field(ge=0, le=2)
    reason: str


def _parse_json(content: str, result_type):
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    try:
        return result_type.model_validate_json(content)
    except Exception:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge 未返回 JSON：{content[:300]}")
        return result_type.model_validate(json.loads(match.group(0)))


class LLMBehaviorJudge:
    """使用已配置 DeepSeek 的最小结构化 Judge，不传入轨迹、chunk 或真实用户历史。"""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def task_success(self, case: AgentEvalCase, output: AgentEvalOutput) -> TaskJudgeResult:
        payload = {
            "turns": case.turns,
            "terminal_behavior": case.expectation.terminal_behavior,
            "answer_criteria": case.expectation.answer_criteria,
            "actual": {
                "final_answer": output.final_answer,
                "needs_clarification": output.needs_clarification,
                "clarification_questions": output.clarification_questions,
            },
        }
        response = self.llm.invoke([
            SystemMessage(content=(
                "你是医疗 Agent 行为评测员。只依据给定冻结用例和实际 Agent 输出评分，"
                "不要补充医学常识，不要推断未提供的检索内容。回答是否完成预期终态与要求："
                "完成为1，否则为0。仅输出 JSON：{\"task_success\":0或1,\"reason\":\"简短中文理由\"}。"
            )),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        return _parse_json(str(response.content), TaskJudgeResult)

    def planning_quality(self, case: AgentEvalCase, output: AgentEvalOutput) -> PlanningJudgeResult | None:
        # 明确追问且未进入规划时没有可评分的规划行为。
        if case.expectation.terminal_behavior == "clarification" and not output.subqueries and not output.rewritten_query:
            return None
        payload = {
            "turns": case.turns,
            "expectation": {
                "need_rewrite": case.expectation.need_rewrite,
                "need_decompose": case.expectation.need_decompose,
                "min_subqueries": case.expectation.min_subqueries,
                "max_subqueries": case.expectation.max_subqueries,
                "expected_topics": case.expectation.expected_topics,
            },
            "actual": {
                "rewritten_query": output.rewritten_query,
                "subqueries": output.subqueries,
            },
        }
        response = self.llm.invoke([
            SystemMessage(content=(
                "你是 Agent 检索规划评测员。只评价改写、是否拆解、子查询覆盖与冗余；"
                "不要用字符串完全匹配。0=无效或明显改变/遗漏核心意图，1=可接受但有冗余或轻微遗漏，"
                "2=合理、完整且精简。仅输出 JSON：{\"planning_quality\":0/1/2,\"reason\":\"简短中文理由\"}。"
            )),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        return _parse_json(str(response.content), PlanningJudgeResult)


def trajectory_tool_correctness(case: AgentEvalCase, output: AgentEvalOutput) -> dict[str, Any]:
    """不依赖 LLM 的关键行为约束检查，允许并行子查询顺序不同。"""
    expectation = case.expectation
    tools = set()
    if output.db_search_count:
        tools.add("database_search")
    if output.web_search_count:
        tools.add("web_search")
    terminal_pass = output.needs_clarification == (expectation.terminal_behavior == "clarification")
    if expectation.terminal_behavior == "clarification":
        terminal_pass = terminal_pass and bool(output.clarification_questions)
    subquery_count = len(output.subqueries)
    decompose_rule_pass = (
        (expectation.min_subqueries is None or subquery_count >= expectation.min_subqueries)
        and (expectation.max_subqueries is None or subquery_count <= expectation.max_subqueries)
    )
    tool_rule_pass = set(expectation.required_tools).issubset(tools) and not (set(expectation.forbidden_tools) & tools)
    web_rule_pass = {
        "required": output.web_search_count > 0,
        "forbidden": output.web_search_count == 0,
        "optional": True,
    }[expectation.web_policy]
    retry_rule_pass = expectation.max_retry_count is None or output.retry_count <= expectation.max_retry_count
    return {
        "trajectory_correct": int(all((terminal_pass, decompose_rule_pass, tool_rule_pass, web_rule_pass, retry_rule_pass))),
        "terminal_rule_pass": terminal_pass,
        "decompose_rule_pass": decompose_rule_pass,
        "tool_rule_pass": tool_rule_pass,
        "web_rule_pass": web_rule_pass,
        "retry_rule_pass": retry_rule_pass,
        "actual_tools": sorted(tools),
    }


def efficiency_record(case: AgentEvalCase, output: AgentEvalOutput) -> dict[str, Any]:
    """只报告真实计数与计时；Provider 未返回 token usage 时保留 null。"""
    return {
        "case_id": case.case_id,
        "turn_count": len(case.turns),
        "subquery_count": len(output.subqueries),
        "retrieval_route_count": sum(len(channels) for channels in output.selected_channels),
        "db_search_count": output.db_search_count,
        "web_search_count": output.web_search_count,
        "retry_count": output.retry_count,
        "llm_call_count": output.llm_call_count,
        "tool_call_count": output.tool_call_count,
        "latency_ms": output.latency_ms,
        "token_usage": output.token_usage,
    }


def aggregate_efficiency(records: list[dict[str, Any]]) -> dict[str, Any]:
    def average(field: str) -> float:
        return mean([float(row[field]) for row in records]) if records else 0.0

    latency = sorted(float(row["latency_ms"]) for row in records)
    def percentile(percent: float) -> float:
        if not latency:
            return 0.0
        index = min(len(latency) - 1, int(round((len(latency) - 1) * percent)))
        return latency[index]

    real_tokens = [row["token_usage"] for row in records if row["token_usage"] is not None]
    return {
        "case_count": len(records),
        "avg_subqueries": average("subquery_count"),
        "avg_db_searches": average("db_search_count"),
        "avg_web_searches": average("web_search_count"),
        "avg_retries": average("retry_count"),
        "latency": {"p50_ms": percentile(0.5), "p95_ms": percentile(0.95)},
        "token_usage": {"available_count": len(real_tokens), "mean": mean(real_tokens) if real_tokens else None},
    }
