"""Stage 4 冻结 Agent Eval Set 与运行输出模型。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AgentExpectation(BaseModel):
    """一条行为用例的人工冻结预期，不描述或替代检索 Gold。"""

    terminal_behavior: Literal["answer", "clarification"]
    need_rewrite: bool | None = None
    need_decompose: bool | None = None
    min_subqueries: int | None = Field(default=None, ge=1, le=3)
    max_subqueries: int | None = Field(default=None, ge=1, le=3)
    expected_topics: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    web_policy: Literal["required", "forbidden", "optional"] = "optional"
    max_retry_count: int | None = Field(default=None, ge=0)
    answer_criteria: list[str] = Field(default_factory=list)

    @field_validator("max_subqueries")
    @classmethod
    def _range_is_ordered(cls, value: int | None, info):
        minimum = info.data.get("min_subqueries")
        if value is not None and minimum is not None and value < minimum:
            raise ValueError("max_subqueries 不能小于 min_subqueries")
        return value


class AgentEvalCase(BaseModel):
    case_id: str
    primary_category: str
    turns: list[str] = Field(min_length=1)
    expectation: AgentExpectation
    note: str = ""

    @field_validator("turns")
    @classmethod
    def _turns_are_not_blank(cls, turns: list[str]) -> list[str]:
        if any(not turn.strip() for turn in turns):
            raise ValueError("turns 不允许包含空字符串")
        return turns


class AgentEvalOutput(BaseModel):
    """每个 Case 的实际运行输出；多轮统计为同一会话的累计值。"""

    case_id: str
    final_answer: str = ""
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    rewritten_query: str = ""
    subqueries: list[str] = Field(default_factory=list)
    selected_channels: list[list[str]] = Field(default_factory=list)
    db_search_count: int = 0
    web_search_count: int = 0
    retry_count: int = 0
    latency_ms: float = 0.0
    trace_id: str | None = None
    trajectory_id: str | None = None
    llm_call_count: int | None = None
    tool_call_count: int | None = None
    token_usage: int | None = None
