from __future__ import annotations

import json
import time
import importlib

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from MedicalRag.agent.SearchGraph import (
    SearchGraph, invoke_with_timeout, merge_web_tool_responses, parse_web_search_response, retrieve,
)
from MedicalRag.agent.MedicalAgent import is_general_knowledge_request, search_one
from MedicalRag.agent.tools import AgentTools
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.retrieval.service import OnlineRetrievalResult
from MedicalRag.retrieval.models import RetrievalPlan, RetrievalTrace
from MedicalRag.retrieval.policy import AgentPlannerPolicy, RetrievalPlannerDecision
from MedicalRag.retrieval.reranker import RerankResult
from MedicalRag.config.models import SearchRequest


class FakePowerModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content="测试回答")


class FakeRetrievalService:
    def __init__(self):
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return OnlineRetrievalResult(
            documents=[Document(page_content="证据", metadata={"pk": "pk-1"})],
            selected_channels=("summary_dense", "text_sparse"),
            planner_error=None,
            candidate_count=30,
            planner_latency_ms=1.0,
            retrieval_latency_ms=2.0,
            reranker_latency_ms=3.0,
        )


def test_retrieve_node_uses_stage2_service_and_records_provenance():
    service = FakeRetrievalService()
    state = {"query": "高血压症状", "docs": [], "node_events": []}
    result = retrieve(state, service, show_debug=False)
    assert service.queries == ["高血压症状"]
    assert [doc.page_content for doc in result["docs"]] == ["证据"]
    assert result["retrieval_info"] == {
        "selected_channels": ["summary_dense", "text_sparse"],
        "planner_error": None,
        "candidate_count": 30,
        "planner_latency_ms": 1.0,
        "retrieval_latency_ms": 2.0,
        "reranker_latency_ms": 3.0,
        "db_search_count": 1,
    }
    assert result["node_events"] == ["retrieval_planner", "retrieval_executor", "reranker"]


def test_search_graph_has_no_direct_database_toolnode(monkeypatch):
    import MedicalRag.agent.SearchGraph as graph_module

    config = ConfigLoader().config.model_copy(deep=True)
    config.agent.network_search_enabled = False
    config.agent.mode = "fast"
    monkeypatch.setattr(graph_module, "create_llm_client", lambda _config: FakePowerModel())
    service = FakeRetrievalService()
    graph = SearchGraph(config, FakePowerModel(), retrieval_service=service)
    graph.build_search_graph()
    assert not hasattr(graph, "db_tool_node")
    assert not hasattr(graph, "db_search_tool")
    assert graph.retrieval_service is service


def test_general_knowledge_boundary_does_not_require_clarification():
    assert is_general_knowledge_request("血压老是高高的，平时吃东西该咋管？")
    assert is_general_knowledge_request("165/105这档算严重不？")
    assert is_general_knowledge_request("今天广州的空气质量是否适合户外运动？")
    assert not is_general_knowledge_request("我这两天头晕，是不是血压有问题？")
    assert not is_general_knowledge_request("老人突然血压不稳应该怎么处理？")


def test_failed_parallel_subquery_isolated_from_other_send_branches():
    class FailingGraph:
        class Config:
            class Agent:
                max_attempts = 1
            agent = Agent()
        config = Config()

        def run(self, _state):
            raise TimeoutError("远端超时")

    output = search_one({"query": "子查询", "subquery_id": 2}, FailingGraph())
    result = output["sub_query_results"][0]
    assert result["query"] == "子查询"
    assert result["subquery_error"] == "TimeoutError"
    assert result["final"] == ""


def test_agent_planner_uses_schema_only_decision_without_database_tool():
    captured = {}

    class PlannerModel:
        def bind_tools(self, tools):
            captured["tools"] = tools
            return self

        def invoke(self, _messages):
            return AIMessage(content="", tool_calls=[{
                "name": "RetrievalPlannerDecision",
                "args": {"selected_channels": ["summary_dense", "text_sparse"]},
                "id": "plan-1",
            }])

    config = ConfigLoader().config
    policy = AgentPlannerPolicy(config, PlannerModel())
    plan = policy.plan("高血压症状")
    assert captured["tools"] == [RetrievalPlannerDecision]
    assert plan.selected_channels == ("summary_dense", "text_sparse")
    assert not hasattr(policy, "database_search_tool")


def test_medical_agent_graph_preserves_semantic_topology(monkeypatch):
    agent_module = importlib.import_module("MedicalRag.agent.MedicalAgent")

    class FakeNormalModel:
        def invoke(self, _messages):
            return AIMessage(content="测试")

    class FakeSearchGraph:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(agent_module, "create_llm_client", lambda _config: FakeNormalModel())
    monkeypatch.setattr(agent_module, "SearchGraph", FakeSearchGraph)
    agent = agent_module.MedicalAgent(ConfigLoader().config, FakeNormalModel())
    edges = {
        (edge.source, edge.target, edge.data, edge.conditional)
        for edge in agent.app.get_graph().edges
    }
    assert ("__start__", "clarification", "ask", True) in edges
    assert ("__start__", "check_update_background", None, True) in edges
    assert ("clarification", "__end__", "ask", True) in edges
    assert ("clarification", "background_update", "pass", True) in edges
    assert ("background_update", "split_query", None, False) in edges
    assert ("check_update_background", "split_query", None, False) in edges
    assert ("split_query", "search_one", None, True) in edges
    assert ("search_one", "gather_answer", None, False) in edges
    assert ("gather_answer", "__end__", None, False) in edges


def test_search_graph_preserves_semantic_topology(monkeypatch):
    import MedicalRag.agent.SearchGraph as graph_module

    config = ConfigLoader().config.model_copy(deep=True)
    config.agent.network_search_enabled = True
    config.agent.mode = "analysis"
    monkeypatch.setattr(graph_module, "create_llm_client", lambda _config: FakePowerModel())
    graph = SearchGraph(config, FakePowerModel(), retrieval_service=FakeRetrievalService())
    graph.build_search_graph()
    edges = {
        (edge.source, edge.target, edge.data, edge.conditional)
        for edge in graph.search_graph.get_graph().edges
    }
    assert ("__start__", "retrieve", None, False) in edges
    assert ("retrieve", "web_search", None, False) in edges
    assert ("web_search", "rag", None, False) in edges
    assert ("rag", "judge", None, False) in edges
    assert ("judge", "finish_success", "pass", True) in edges
    assert ("judge", "rag", "retry", True) in edges
    assert ("judge", "finish_fail", "fail", True) in edges


def test_online_service_composes_planner_executor_and_reranker():
    import asyncio
    from MedicalRag.retrieval.service import OnlineRetrievalService

    config = ConfigLoader().config
    plan = RetrievalPlan("agent_planner", ("summary_dense",), SearchRequest(query="问题"))

    class Planner:
        last_planner_latency_ms = 1.5

        def plan(self, query):
            assert query == "问题"
            return plan

    class Executor:
        def execute(self, received_plan):
            assert received_plan is plan
            return RetrievalTrace(plan, [Document(page_content="第一"), Document(page_content="第二")], False, 2.5)

    class Reranker:
        async def rerank(self, query, documents, top_n):
            assert (query, documents, top_n) == ("问题", ["第一", "第二"], 5)
            return [RerankResult(index=1, score=0.9), RerankResult(index=0, score=0.1)]

    service = OnlineRetrievalService(config, FakePowerModel(), planner=Planner(), executor=Executor(), reranker=Reranker())
    result = asyncio.run(service.retrieve_async("问题"))
    assert [document.page_content for document in result.documents] == ["第二", "第一"]
    assert result.candidate_count == 2


def test_web_tool_configuration_error_is_structured_json():
    config = ConfigLoader().config
    tools = AgentTools(config)

    def missing_key(_query, _count):
        raise KeyError("TAVILY_API_KEY")

    tools.register_websearch(missing_key)
    payload = json.loads(tools.make_web_search_tool().invoke({"query": "测试问题"}))
    assert payload == {
        "ok": False,
        "documents": [],
        "error": {
            "code": "configuration_error",
            "message": "联网搜索服务未配置：缺少 TAVILY_API_KEY。",
        },
    }
    documents, error = parse_web_search_response(json.dumps(payload, ensure_ascii=False))
    assert documents == []
    assert error == payload["error"]


def test_web_tool_response_parsing_keeps_legacy_success_format():
    documents, error = parse_web_search_response(json.dumps([
        {"page_content": "联网证据", "metadata": {"source": "test"}},
    ], ensure_ascii=False))
    assert error is None
    assert [document.page_content for document in documents] == ["联网证据"]


def test_web_tool_timeout_is_structured_json(monkeypatch):
    tools_module = importlib.import_module("MedicalRag.agent.tools.AgentTools")

    config = ConfigLoader().config
    tools = AgentTools(config)
    monkeypatch.setattr(tools_module, "WEB_SEARCH_TIMEOUT_SECONDS", 0.01)
    tools.register_websearch(lambda _query, _count: time.sleep(1))
    payload = json.loads(tools.make_web_search_tool().invoke({"query": "测试问题"}))
    assert payload["ok"] is False
    assert payload["documents"] == []
    assert payload["error"]["code"] == "provider_timeout"


def test_web_router_call_timeout_has_a_hard_upper_bound():
    with pytest.raises(TimeoutError):
        invoke_with_timeout(lambda: time.sleep(1), 0.01)


def test_multiple_web_tool_results_are_merged_and_deduplicated():
    def payload(url, content):
        return json.dumps({
            "ok": True,
            "documents": [{"page_content": content, "metadata": {"url": url}}],
            "error": None,
        }, ensure_ascii=False)

    messages = [
        ToolMessage(content=payload("https://one", "中文证据"), tool_call_id="one"),
        ToolMessage(content=payload("https://two", "英文证据"), tool_call_id="two"),
        ToolMessage(content=payload("https://one", "中文证据"), tool_call_id="three"),
    ]
    documents, errors, success_count = merge_web_tool_responses(messages)
    assert [document.page_content for document in documents] == ["中文证据", "英文证据"]
    assert errors == []
    assert success_count == 3
