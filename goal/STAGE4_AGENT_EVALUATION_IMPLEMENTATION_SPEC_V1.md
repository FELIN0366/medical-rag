# Stage 4 Implementation Spec v1
# Stage 2 Retrieval Runtime Integration + Local Agent Trajectory Evaluation Harness

> **目标仓库（唯一允许修改）**：`FELIN0366/medical-rag`  
> **执行环境**：WSL Ubuntu  
> **Conda 环境**：`rag`  
> **核心原则**：Stage 4 不再增加新的 Agent 能力，只完成**真实在线主链收口 + 本地 Agent Evaluation Harness**。  
> **本阶段只做四件事**：  
> 1. 将 Stage 2 已验证的 `Planner → UnifiedRetrievalExecutor → Reranker` 正式接入 `SearchGraph` 在线检索节点；  
> 2. 在 LangGraph 执行层增加**本地 trajectory collector**，不接入、不上传 LangSmith / Langfuse Cloud；  
> 3. 构建并冻结 30 条小规模、高质量 Agent Behavior Eval Dataset；  
> 4. 实现四个 Evaluator：`Task Success / Planning Quality / Trajectory & Tool Correctness / Efficiency`。  
>
> **明确不做**：LangSmith Cloud、Langfuse Cloud、自托管 Observability 平台、Evidence Grade、Citation Verification、Safety Benchmark、Memory Benchmark、Reranker Model Zoo、更大 Retrieval Eval、更多 Corpus、更多 Planner Action、Fixed RAG vs Full Agent 消融。  
> **完成后停止扩展项目，进入面试材料整理。**

---

# 0. 本版冻结决策

## 0.1 Stage 4 不增加新功能

Stage 1 已完成：

```text
QA / PDF / HTML / Markdown
→ Parser / Chunk
→ summary_dense / text_dense / BM25
→ Milvus
```

Stage 2 已完成：

```text
Fixed Hybrid
+ Dynamic Retrieval Planner
+ qwen3.7-text-rerank
+ Retrieval Evaluation
```

Stage 4 只做：

```text
已验证 Retrieval Stack
→ 接管真实 SearchGraph Runtime

MedicalAgent / SearchGraph
→ Local Trace Collector

30 Agent Behavior Cases
→ Local Agent Evaluation Harness
```

禁止继续增加新的业务节点。

---

## 0.2 Stage 4 评价“Agent 是否做对决策”

Stage 2 已评价：

```text
Candidate Recall@30
Recall@5
MRR
nDCG@5
Reranker Gain
routes/query
```

Stage 4 不重复这些 Retrieval Metrics。

Stage 4 只评价：

```text
最终任务是否完成
Planning 是否合理
Tool / Trajectory 是否合理
执行是否高效
```

即：

```text
Retrieval Evaluation
→ 知识找得准不准

Agent Evaluation
→ Agent 做事做得对不对
```

---

## 0.3 本地 Trace，不依赖云端 Observability

Stage 4 不接入：

```text
LangSmith Cloud
Langfuse Cloud
OpenAI Trace Cloud
```

也不在本阶段部署：

```text
self-hosted Langfuse
self-hosted LangSmith
Kubernetes observability stack
```

原因不是“Trace 算法不需要平台”，而是当前目标是：

```text
可重复 Agent Evaluation
+
节点级行为归因
+
医疗上下文不额外上传第三方 Observability 平台
```

因此本阶段采用：

```text
LangGraph Runtime
      ↓
LocalTraceCollector
      ↓
artifacts/stage4/trajectory.jsonl
      ↓
Code Evaluator + LLM Judge
      ↓
metrics.json / failure_cases.jsonl
```

Observability 平台不是 Agent Evaluation 的必要依赖。

---

## 0.4 “本地评测”与模型 API 的边界

本阶段的**trajectory、retrieved document metadata、tool decision、failure case、evaluation artifact 全部保存在本地**，不上传 LangSmith / Langfuse。

但当前项目本身仍使用：

```text
DashScope Embedding
DashScope Reranker
DeepSeek LLM
```

因此不要把 Stage 4 描述成：

```text
100% air-gapped / 全离线
```

正确表述是：

> **Agent trajectory 与评测数据持久化在本地，不额外上传第三方 observability 平台。**

`Task Success / Planning Quality` 如果继续使用 `deepseek-flash` 作为 Judge，只允许发送：

```text
冻结的 Eval Case
最终 Agent 输出
必要的 reference criteria
```

禁止把以下内容作为 Judge 输入：

```text
真实用户历史
完整 production trace
完整 retrieved chunks
API Key / 环境变量
```

如果未来要求严格零外部 Judge，可替换成本地 Judge；这不属于 Stage 4 P0。

---

## 0.5 不做 Legacy Runtime Compatibility

当前真实在线 `SearchGraph` 的旧本地检索路径是：

```text
llm_db_search
→ database_search ToolNode
→ AgentTools.database_search()
→ get_kb(...).search(SearchRequest)
```

Stage 4 必须替换为：

```text
AgentPlannerPolicy
→ UnifiedRetrievalExecutor
→ optional RRF
→ qwen3.7-text-rerank
→ Top-5
```

新在线主链通过：

```text
unit tests
+ SearchGraph smoke
+ MedicalAgent smoke
+ Local Trace smoke
```

后：

- 删除 `SearchGraph` 对旧 DB ToolNode **执行路径**的依赖；
- 删除只服务旧在线 DB 执行的重复逻辑，前提是 code search 确认无有效引用；
- 不保留“新旧两套在线检索器可切换”的兼容开关；
- Stage 2 的 `SearchRequest / AgentPlannerPolicy / UnifiedRetrievalExecutor / DashScopeReranker` 为唯一检索算法主链。

注意：

`database_search` 的 Tool Schema 如果仍被 `AgentPlannerPolicy` 用于结构化 Tool Call，可保留其 schema 能力；但在线 `SearchGraph` 不得再通过 ToolNode 直接执行旧 `KnowledgeBase.search()`。

---

# 1. Codex 开始前必须只读检查

## 1.1 先检查当前未提交改动

当前 Codex 已产生但尚未验证的本地改动包括：

```text
Stage 2 online service layer 雏形
SearchGraph 接入该 service 的雏形
provenance / execution counter
Stage 4 local eval models
30 条行为用例草案
可能存在的 LangSmith traceable import
```

开始前必须：

```bash
git status --short
git diff --stat
git diff
```

要求：

- 不要直接 `git reset --hard`；
- 先审计哪些改动符合本 Spec；
- 符合的新 Runtime / Dataset 代码继续复用并补完；
- 删除所有 LangSmith / Langfuse 依赖、import、环境变量读取和远程 tracing 逻辑；
- 不保留无实际用途的 `traceable` 装饰器。

---

## 1.2 我方仓库必须阅读

```text
README.md
environment.yml

src/MedicalRag/config/models.py
src/MedicalRag/config/app_config.yaml
src/MedicalRag/config/loader.py

src/MedicalRag/agent/MedicalAgent.py
src/MedicalRag/agent/SearchGraph.py
src/MedicalRag/agent/tools/AgentTools.py
src/MedicalRag/prompts/templates.py

src/MedicalRag/retrieval/models.py
src/MedicalRag/retrieval/policy.py
src/MedicalRag/retrieval/executor.py
src/MedicalRag/retrieval/reranker.py

src/MedicalRag/evaluation/retrieval_dataset.py
src/MedicalRag/evaluation/retrieval_metrics.py
src/MedicalRag/evaluation/retrieval_runner.py

src/MedicalRag/api/app.py

scripts/09_prepare_stage2_eval.py
scripts/10_eval_stage2_retrieval.py

artifacts/stage2/metrics.json
artifacts/stage2/reranker_analysis.json
artifacts/stage2/planner_analysis.json

docs/stage2_retrieval_reranker_planner_implementation.md
docs/stage2_retrieval_reranker_planner_interview.md
```

如果当前工作区已新增：

```text
src/MedicalRag/retrieval/service.py
src/MedicalRag/agent_evaluation/*
data/eval/agent_behavior_cases.jsonl
```

也必须先完整阅读再修改。

---

## 1.3 必须确认当前在线 Agent 真实调用链

重点核对：

```text
MedicalAgent.answer()
MedicalAgent.build_graph()
route_to_subgraphs()
search_one()

SearchGraph.build_search_graph()
llm_db_search() 或新 retrieve node
llm_network_search()
rag()
judge()

AgentTools.make_database_search_tool()
```

修改前先画出当前真实调用关系。

---

## 1.4 必须确认 Stage 2 已验证主链

重点核对：

```text
AgentPlannerPolicy.plan()
normalize_plan()
UnifiedRetrievalExecutor.execute()
DashScopeReranker.rerank()
RetrievalEvaluationRunner.run()
```

确认冻结参数仍为：

```text
per_route_k = 50
rrf_k       = 60
candidate_k = 30
final_k     = 5
```

确认正式模型仍为：

```text
Dense: qwen3.7-text-embedding-flash
Reranker: qwen3.7-text-rerank
```

Stage 4 不改变这些参数。

---

# 2. 执行环境

所有命令：

```bash
cd <FELIN0366/medical-rag>
conda activate rag
```

开始前报告：

```bash
pwd
git status --short
git rev-parse HEAD
which python
python --version
conda env list
docker ps --filter name=milvus-standalone
```

检查已有模型 API 环境变量，只输出是否存在，不输出值：

```bash
python - <<'PY'
import os
for key in [
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_WORKSPACE_ID",
    "DEEPSEEK_API_KEY",
]:
    print(key, bool(os.getenv(key)))
PY
```

**不需要 `LANGSMITH_API_KEY`。**

约束：

- 不创建新 Conda 环境；
- 不升级 torch / CUDA / Milvus 等无关依赖；
- 不为 Stage 4 安装 LangSmith / Langfuse / AgentEvals；
- `environment.yml` 中即使已有 `langsmith` 依赖，本阶段代码不得依赖其远程 tracing；
- 如果现有项目其他功能仍依赖该包，不强制从环境删除；只删除 Stage 4 新增的 LangSmith 代码依赖；
- 不引入新的 Agent 框架。

---

# 3. Stage 4 最终在线 Pipeline

```text
User
  ↓
MedicalAgent
  │
  ├── ask / clarification
  ├── background update
  ├── rewrite / decompose
  └── Send(sub-query)
          ↓
       SearchGraph
          ↓
   AgentPlannerPolicy
          ↓
selected S / T / B
          ↓
UnifiedRetrievalExecutor
          ↓
single route / RRF
          ↓
Candidate Top-30
          ↓
qwen3.7-text-rerank
          ↓
Final Top-5
          ↓
existing Web Router
          ↓
optional Web Search
          ↓
existing RAG Generate
          ↓
existing Judge / bounded retry
          ↓
sub-answer
  │
  └── gather_answer
          ↓
      Final Answer
```

Stage 4 不新增：

```text
Evidence Grade
Citation Verify
Safety Node
Memory Node
```

---

# 4. 在线 Retrieval Service

如果当前 Codex 已新增：

```text
src/MedicalRag/retrieval/service.py
```

优先在现有基础上完成，不重新造第二个 service。

职责只能是：

```text
query
↓
AgentPlannerPolicy.plan()
↓
UnifiedRetrievalExecutor.execute()
↓
DashScopeReranker
↓
Top-5 Documents
+ execution metadata
```

建议统一返回：

```python
@dataclass
class OnlineRetrievalResult:
    documents: list[Document]
    selected_channels: tuple[str, ...]
    planner_error: str | None

    candidate_count: int
    candidate_pks: list[str]
    final_pks: list[str]

    planner_latency_ms: float
    retrieval_latency_ms: float
    reranker_latency_ms: float
```

Service 中禁止复制：

```text
Milvus search
RRF
embedding
planner prompt
reranker request/response logic
```

这些必须继续复用 Stage 2 代码。

---

# 5. 同步 / 异步边界

当前 `MedicalAgent` / `SearchGraph` 主节点以同步调用为主，FastAPI 阻塞接口通过线程池执行。

要求：

- 保留现有同步 `SearchGraph` 主体；
- Online Retrieval Service 提供兼容现有节点的同步入口；
- Reranker 底层 payload、response parser、retry policy 只能维护一份；
- 若需要同时保留 async eval 与 sync runtime，只增加薄的 sync/async 入口，不复制算法实现。

---

# 6. SearchGraph 替换本地检索节点

当前旧逻辑：

```text
db_search
→ llm_db_search()
→ ToolNode(database_search)
```

最终改为一个唯一在线 Retrieval Node，例如：

```text
retrieve
```

实现：

```text
query
↓
OnlineRetrievalService.retrieve()
↓
Top-5 Documents
↓
state["docs"]
```

同时写入：

```text
state["retrieval_info"]
```

至少包含：

```text
selected_channels
planner_error
candidate_count
candidate_pks
final_pks
planner_latency_ms
retrieval_latency_ms
reranker_latency_ms
```

---

# 7. SearchGraph State 扩展

在 `SearchMessagesState` 增加最小观测字段：

```python
retrieval_info: dict
web_search_count: int
judge_retry_count: int
trace_id: str | None
subquery_id: str | None
```

这些只用于：

```text
Local Trace
Offline Evaluator
Failure Analysis
```

不引入复杂 telemetry framework。

---

# 8. Web Search 与 Judge 保持现有逻辑

现有：

```text
llm_network_search()
rag()
judge()
```

继续保留。

Stage 4 只记录：

```text
need_search
search_query
是否实际调用 web_search
web_search_count
judge_result
retry_count_used
```

不新增 Evidence Judge / Citation Judge。

---

# 9. Retrieval Metadata 必须完整

在线 Stage 2 Retrieval 的 `SearchRequest.output_fields` 至少包含：

```text
pk
text
summary
document
source
source_name
doc_id
chunk_id
department
title
section_path
page
```

如果当前默认只返回：

```text
pk / text / summary / document
```

本阶段补齐。

这是 provenance / failure analysis 所需，不是新增 Retrieval 功能。

---

# 10. Local Trace Collector

新增：

```text
src/MedicalRag/agent_evaluation/trace.py
```

实现一个轻量、线程安全的：

```text
LocalTraceCollector
```

要求：

- 纯本地内存 + JSONL；
- 不发送网络请求；
- 不依赖 LangSmith / Langfuse；
- 支持 LangGraph `Send` 并行子任务；
- 同一 Case / Session 可以聚合同一 trace；
- Eval Runner 可注入；
- 正常 API Runtime 默认可以不持久化 Trace。

---

# 11. Trace Event Schema

建议：

```python
class TraceEvent(BaseModel):
    trace_id: str
    case_id: str | None = None
    session_id: str

    turn_index: int
    subquery_id: str | None = None

    node: str
    event: str

    timestamp_ms: float
    duration_ms: float | None = None

    payload: dict = Field(default_factory=dict)
```

其中：

```text
trace_id
```

标识一次完整 Eval Case；

```text
session_id
```

与现有 MedicalAgent session 对齐；

```text
subquery_id
```

用于并行 `Send` 后区分各 SearchGraph。

不要依赖单一顺序索引来表达并行关系。

---

# 12. Trace 节点名称冻结

只记录 Agent 决策关键节点：

```text
medical_agent
clarification
background_update
split_query
search_one

retrieval_planner
retrieval_executor
reranker

web_router
web_search

rag_generate
judge
gather_answer
```

不要记录每个 Python helper。

---

# 13. 各 Trace 节点记录什么

## 13.1 clarification

```text
need_ask
questions
```

## 13.2 split_query

```text
need_split
rewrite_query
sub_queries
```

## 13.3 retrieval_planner

```text
selected_channels
planner_error
planner_latency_ms
```

## 13.4 retrieval_executor

```text
candidate_count
candidate_pks
retrieval_latency_ms
```

## 13.5 reranker

```text
input_candidate_count
final_pks
reranker_latency_ms
```

## 13.6 web_router

```text
need_search
search_query
```

## 13.7 web_search

```text
called
search_query
result_count
```

## 13.8 judge

```text
judge_result
retry_count_used
```

## 13.9 gather_answer

```text
final_answer
```

---

# 14. Trace 内容保持紧凑

默认不把以下内容完整写入 trajectory：

```text
完整 Top-30 文档正文
完整 prompt
完整模型内部消息
完整生产用户历史
```

检索阶段默认记录：

```text
pk
source
source_name
title
section_path
```

以及必要的：

```text
candidate_pks
final_pks
```

如 failure analysis 需要正文，可通过 `pk` 回查冻结 Collection；不要把整个 Corpus 复制进 trace artifact。

---

# 15. Trace 注入方式

`MedicalAgent` 和 `SearchGraph` 增加可选依赖：

```python
trace_collector: LocalTraceCollector | None = None
```

正常在线 API：

```text
trace_collector = None
```

默认不持久化真实用户 trajectory。

Stage 4 Eval Runner：

```text
trace_collector = LocalTraceCollector(...)
```

只对冻结 Eval Dataset 记录本地 trace。

不要在生产接口中默认全量记录医疗用户输入。

---

# 16. Local Trace 输出

正式 Eval 生成：

```text
artifacts/stage4/trajectory.jsonl
```

每一行是一个 `TraceEvent`。

另外生成：

```text
artifacts/stage4/run_summaries.jsonl
```

每 Case 一行，聚合关键行为。

---

# 17. Stage 4 Eval Dataset 规模

最终冻结：

```text
30 cases
```

分布：

| Primary Category | Count |
| --- | ---: |
| simple_single_hop | 5 |
| rewrite_needed | 5 |
| decompose_multi_intent | 5 |
| clarification_needed | 4 |
| kb_sufficient_no_web | 4 |
| kb_insufficient_web | 3 |
| multi_turn_context | 4 |
| **Total** | **30** |

一条 Case 可以同时具备次级属性，但只保留一个 `primary_category`。

---

# 18. Agent Eval Dataset Schema

新增 / 完善：

```text
src/MedicalRag/agent_evaluation/models.py
```

最终：

```python
class AgentExpectation(BaseModel):
    terminal_behavior: Literal["answer", "clarification"]

    need_rewrite: bool | None = None
    need_decompose: bool | None = None

    min_subqueries: int | None = None
    max_subqueries: int | None = None

    expected_topics: list[str] = []

    required_tools: list[str] = []
    forbidden_tools: list[str] = []

    web_policy: Literal["required", "forbidden", "optional"] = "optional"

    max_retry_count: int | None = None

    answer_criteria: list[str] = []


class AgentEvalCase(BaseModel):
    case_id: str
    primary_category: str
    turns: list[str]
    expectation: AgentExpectation
    note: str = ""
```

---

# 19. 为什么统一使用 turns

单轮：

```json
{"turns": ["高血压一般有哪些症状？"]}
```

多轮：

```json
{
  "turns": [
    "我最近血压偏高。",
    "如果吃药后一直干咳，可能和药有关系吗？",
    "那我还需要做哪些检查？"
  ]
}
```

同一个 Runner 处理：

```text
single-turn
+
multi-turn
```

不建两套 schema。

---

# 20. Dataset 构造原则

30 条数据必须：

- 人工可读；
- 行为预期明确；
- 尽量建立在当前 Corpus 能力上；
- 不写回 Milvus；
- 不把主题相似自动当 Gold；
- 不为了凑数量生成含糊 expected behavior；
- 不包含真实用户隐私或真实医疗记录。

允许 LLM 生成候选问题，但最终 `AgentExpectation` 必须显式冻结，不把自动标签直接当真值。

---

# 21. 各类 Case 约束

## 21.1 Simple Single-hop：5 条

```text
terminal_behavior = answer
need_decompose = false
max_subqueries = 1
web_policy = forbidden
```

测试：

```text
不应过度拆分
不应无意义 Web
```

## 21.2 Rewrite Needed：5 条

输入口语化 / 指代化，但意图单一。

```text
need_rewrite = true
need_decompose = false
max_subqueries = 1
```

`expected_topics` 用于 Judge 判断改写是否保持意图。

## 21.3 Decompose Multi-intent：5 条

至少两个独立 Evidence Need：

```text
need_decompose = true
min_subqueries = 2
max_subqueries = 3
```

必须提供：

```text
expected_topics
```

不要求子查询文本完全一致。

## 21.4 Clarification Needed：4 条

```text
terminal_behavior = clarification
```

评价：

```text
是否确实发起关键追问
追问是否针对信息缺口
```

## 21.5 KB Sufficient / No Web：4 条

选择当前 QA / Literature 明确覆盖的问题：

```text
required_tools = ["database_search"]
web_policy = forbidden
```

## 21.6 KB Insufficient / Web Required：3 条

选择当前冻结 Corpus 明确缺少充分证据的问题：

```text
web_policy = required
```

只测试 Agent 是否正确触发已有 Web 路径，不新增 Web Search 算法。

## 21.7 Multi-turn Context：4 条

每条 2～3 turns，重点验证：

```text
background_info
dialogue history
rewrite/decompose decision
跨轮指代
```

同一 Case 使用同一 Agent instance / session id。

---

# 22. Dataset 文件

最终：

```text
data/eval/agent_behavior_cases.jsonl
```

冻结副本：

```text
artifacts/stage4/agent_eval_dataset.jsonl
```

两者内容一致时记录 hash。

Eval Dataset 不进入 Milvus Corpus。

---

# 23. 四个 Evaluator

最终只实现：

```text
1. Task Success
2. Planning Quality
3. Trajectory / Tool Correctness
4. Efficiency
```

不增加：

```text
Retrieval Recall
Faithfulness
Citation Score
Safety Score
Medical Rubric 总分
```

Stage 4 的重点是 Agent 行为，而不是再造一套 RAG Eval。

---

# 24. Evaluator 1：Task Success

目的：

> 最终行为是否完成 Case 目标。

对于：

```text
terminal_behavior = answer
```

输入：

```text
all turns
final answer
answer_criteria
```

使用结构化 LLM Judge：

```text
task_success ∈ {0,1}
reason
```

对于：

```text
terminal_behavior = clarification
```

评价：

```text
是否正确进入 clarification
追问是否针对关键信息缺口
```

仍输出 0/1。

---

# 25. Evaluator 2：Planning Quality

只评价：

```text
rewrite
decompose
sub_queries
```

输入：

```text
original turns
need_rewrite expectation
need_decompose expectation
actual rewritten query
actual subqueries
expected_topics
```

LLM Judge 输出：

```text
planning_quality ∈ {0,1,2}
```

定义：

```text
0 = invalid
    错误拆解 / 漏掉核心意图 / 明显改变问题

1 = acceptable
    基本覆盖意图，但存在冗余或轻微遗漏

2 = good
    是否拆解判断合理，覆盖完整，子查询精简且互补
```

Clarification Case 若未进入 planning：

```text
planning_quality = N/A
```

统计时必须同时报告：

```text
applicable_count
```

---

# 26. Planning Judge 不做字符串完全匹配

禁止：

```text
actual_subquery == reference_subquery
```

Judge 重点看：

```text
necessity
coverage
redundancy
semantic faithfulness
```

---

# 27. LLM Judge 的隐私边界

优先复用当前已配置：

```text
deepseek-flash
```

只用于：

```text
Task Success
Planning Quality
```

Judge 输入禁止包含：

```text
完整 trajectory
retrieved chunk 全文
真实用户历史
production session data
```

只发送：

```text
冻结 Eval Case
必要 expectation
final answer / rewrite / subqueries
```

如果未来要求 Judge 也完全本地化，只替换 Judge Provider；不改变 Harness Schema。

---

# 28. Evaluator 3：Trajectory / Tool Correctness

优先使用纯代码 deterministic evaluator。

检查：

```text
terminal behavior
subquery count range
required tools
forbidden tools
web policy
retry upper bound
```

输出：

```text
trajectory_correct ∈ {0,1}
```

并保存：

```text
terminal_rule_pass
decompose_rule_pass
tool_rule_pass
web_rule_pass
retry_rule_pass
```

---

# 29. Tool / Behavior 映射

本地 Trace 中统一把行为抽象为：

```text
database_search
web_search
clarification
```

其中：

```text
database_search
```

代表：

```text
AgentPlannerPolicy
→ UnifiedRetrievalExecutor
→ Reranker
```

整条本地 Retrieval 调用。

不要因为 Stage 2 内部有：

```text
summary_dense
text_dense
BM25
```

就把它们误当成三个 Agent Tool。

这些是 Retrieval Channel，不是顶层 Tool。

---

# 30. 不使用完整 strict trajectory 作为主评分

因为：

```text
LangGraph Send
```

存在并行子任务。

例如：

```text
Q1 / Q2
```

完成顺序变化不等于行为错误。

所以 Stage 4 主评测只检查：

```text
关键节点是否发生
Tool 是否正确
分解数量是否合理
Web 是否合理
Retry 是否超限
```

不要求完整事件严格等序。

---

# 31. Evaluator 4：Efficiency

完全代码统计，不调用 LLM。

每 Case 记录：

```text
turn_count
subquery_count
retrieval_route_count

db_search_count
web_search_count
retry_count

llm_call_count
tool_call_count

latency_ms
token_usage
```

`token_usage`：

- 如果当前 Provider response / callback 能获得真实值，则记录；
- 无法获得则保存 `null`；
- 禁止通过字符数粗略伪造。

---

# 32. Efficiency 不合成单一总分

不要设计：

```text
Agent Efficiency Score = 87.3
```

只输出真实统计：

```text
Avg DB Calls / Case
Avg Web Calls / Case
Avg Retry / Case
Avg Subqueries / Case
p50 / p95 Latency
Avg Tokens（若真实可用）
```

---

# 33. Agent Evaluation Runtime Output

统一：

```python
class AgentEvalOutput(BaseModel):
    case_id: str
    primary_category: str

    final_answer: str
    needs_clarification: bool
    clarification_questions: list[str]

    rewritten_query: str
    subqueries: list[str]

    selected_channels: list[list[str]]

    db_search_count: int
    web_search_count: int
    retry_count: int

    latency_ms: float
    token_usage: int | None = None

    trace_id: str
```

Multi-turn：

```text
final_answer
```

表示最后一个 turn 的最终结果；行为统计为整个 Case 聚合。

---

# 34. 本地 Trace Summary

每个 Case 从 `trajectory.jsonl` 聚合成：

```text
run_summaries.jsonl
```

每条至少：

```text
case_id
trace_id
session_id
primary_category

turn_count
subquery_count
rewritten_query
subqueries
selected_channels

db_search_count
web_search_count
retry_count

final_answer
needs_clarification

latency_ms
token_usage
```

Evaluator 优先读取：

```text
run_summaries.jsonl
+
agent_eval_dataset.jsonl
```

只有 failure analysis 需要时再读取完整 trajectory。

---

# 35. Stage 4 Dataset 构建脚本

新增 / 完善：

```text
scripts/11_prepare_stage4_agent_eval.py
```

职责：

```text
验证 30 case schema
验证 category 数量
验证 case_id 唯一
验证 turns 非空
验证 expectation 合法
输出冻结副本
输出 dataset report
```

不在每次运行时重新随机生成 Eval Set。

---

# 36. Stage 4 Eval Runner

新增唯一 Runner：

```text
scripts/12_eval_stage4_agent.py
```

流程：

```text
load 30 local cases
↓
对每个 Case 创建干净 MedicalAgent session
↓
注入 LocalTraceCollector
↓
顺序执行 turns
↓
本地保存 trajectory
↓
聚合 AgentEvalOutput / run summary
↓
Task Success Judge
↓
Planning Quality Judge
↓
Deterministic Trajectory Evaluator
↓
Efficiency Aggregation
↓
写 artifacts/stage4
```

正式命令：

```bash
python scripts/12_eval_stage4_agent.py \
  --dataset data/eval/agent_behavior_cases.jsonl \
  --artifact-dir artifacts/stage4
```

可提供：

```text
--smoke 2
```

正式结果必须完整：

```text
30 / 30 cases
```

---

# 37. Local Trace Smoke

正式评测前先跑：

```text
1 simple case
1 multi-turn / decompose case
```

必须验证本地 `trajectory.jsonl` 包含：

```text
medical_agent
split_query
search_one
retrieval_planner
retrieval_executor
reranker
web_router / web_search（若触发）
rag_generate
judge
gather_answer
```

并确认：

```text
同一个 Case 多个 Turn
```

拥有相同：

```text
trace_id / session_id
```

但不同：

```text
turn_index
```

未通过 Local Trace Smoke，不允许直接跑正式 30 条。

---

# 38. 本地 Artifacts

最终生成：

```text
artifacts/stage4/
├── agent_eval_dataset.jsonl
├── dataset_report.json
├── trajectory.jsonl
├── run_summaries.jsonl
├── predictions.jsonl
├── evaluator_results.jsonl
├── metrics.json
├── efficiency.json
├── failure_cases.jsonl
└── config_snapshot.yaml
```

不生成：

```text
langsmith_runs.jsonl
langfuse_runs.jsonl
```

---

# 39. predictions.jsonl

每条至少：

```text
case_id
primary_category
turns

final_answer
needs_clarification
clarification_questions

rewritten_query
subqueries
selected_channels

db_search_count
web_search_count
retry_count

latency_ms
token_usage
trace_id
```

---

# 40. evaluator_results.jsonl

每条：

```text
case_id

task_success
task_success_reason

planning_quality
planning_quality_reason

trajectory_correct
trajectory_checks

efficiency
```

不要只保存平均值。

---

# 41. metrics.json

总体至少：

```text
case_count

task_success_rate

planning_quality:
  applicable_count
  mean
  good_ratio

trajectory_correct_rate

avg_subqueries
avg_db_searches
avg_web_searches
avg_retries

latency:
  p50
  p95

token_usage:
  available_count
  mean
```

同时按：

```text
primary_category
```

分组输出核心指标。

---

# 42. Failure Analysis

只保留少量可解释失败。

分类：

```text
planning_error
unnecessary_decomposition
missing_decomposition
unnecessary_web
missing_web
retry_overuse
task_failure
```

每类最多：

```text
5
```

每条记录：

```text
case_id
trace_id
turns
expected behavior
actual planning
actual tools
final answer
```

不依赖外部 UI，直接通过：

```text
trace_id
```

在 `trajectory.jsonl` 中定位完整轨迹。

---

# 43. 不重复 Stage 2 Retrieval Eval

Stage 4 不重新输出：

```text
Recall@5
MRR
nDCG
Candidate Recall
Reranker Rescue
Reranker Damage
```

这些仍以：

```text
artifacts/stage2/
```

为准。

---

# 44. 测试

新增 / 完善：

```text
tests/test_stage4_runtime_integration.py
tests/test_stage4_agent_evaluation.py
```

---

# 45. Runtime Integration Test

至少验证：

```text
SearchGraph 本地检索经过 AgentPlannerPolicy
SearchGraph 使用 UnifiedRetrievalExecutor
SearchGraph 使用 DashScopeReranker 接口
旧 ToolNode direct DB execution 不再是 Runtime 主链
Top-5 docs 正确进入 state["docs"]
retrieval_info 正确记录
```

全部使用 Fake / Mock，不依赖真实 API。

---

# 46. Local Trace Test

至少验证：

```text
LocalTraceCollector 不发网络请求
同一 case_id / trace_id 可聚合
不同 turn_index 正确
并行 subquery 可通过 subquery_id 区分
事件可写 trajectory.jsonl
run summary 聚合正确
```

---

# 47. Agent Evaluation Test

至少验证：

```text
30 Case Schema 合法
case_id 唯一
category 数量正确

Task Success evaluator structured output
Planning evaluator 0/1/2
Trajectory evaluator deterministic rule
Efficiency aggregation

multi-turn case 使用同一 session
N/A planning 不进入 planning mean

artifact 完整生成
```

---

# 48. API / Agent Session

当前：

```text
/api/agent
/api/agent/stream
```

继续使用现有：

```text
session_id
```

不修改前端协议。

正常真实用户 Runtime：

```text
trace_collector = None
```

Stage 4 Eval Runtime：

```text
trace_collector = LocalTraceCollector
```

不要默认把所有真实医疗对话写入本地 Eval Trace。

---

# 49. 不新增 Observability Platform

Stage 4 明确不实现：

```text
LangSmith
Langfuse
Dashboard
Alert
Production Sampling
Quality Drift Monitor
Cron
Kubernetes
```

面试定位：

> 在 LangGraph execution 层实现轻量级本地 Agent Evaluation Harness，而不是搭建 Observability Platform。

---

# 50. 技术文档

完成后强制生成：

```text
docs/stage4_agent_evaluation_implementation.md
docs/stage4_agent_evaluation_interview.md
```

---

# 51. Implementation 文档

必须记录真实结果：

```text
1. Stage 4 完整 Pipeline
2. Stage 2 Retrieval Stack 如何接管 SearchGraph
3. 删除了哪些旧 Runtime 检索路径
4. LocalTraceCollector 设计
5. Trajectory Event Schema
6. Trace hierarchy
7. Agent Eval Dataset 30 条分布
8. Dataset Schema
9. Task Success
10. Planning Quality
11. Trajectory / Tool Correctness
12. Efficiency
13. 实际运行指标
14. Failure Breakdown
15. 典型本地 Trace
16. 数据隐私边界
17. 已知限制
```

不能写未真实运行的结果。

---

# 52. 面试文档

至少回答：

```text
1. 为什么 Retrieval Eval 后还需要 Agent Eval？
2. Stage 2 与 Stage 4 分别评价什么？
3. 为什么不继续加 Evidence Grade / Citation？
4. 为什么不使用 LangSmith Cloud？
5. Observability Platform 和 Agent Evaluation Harness 有什么区别？
6. MedicalAgent 与 SearchGraph 的双层关系是什么？
7. Stage 2 Retrieval Stack 如何接入真实 SearchGraph？
8. 为什么 Planner / Executor / Reranker 要共用同一在线主链？
9. LocalTraceCollector 如何记录并行 LangGraph trajectory？
10. 为什么只记录核心决策节点？
11. 为什么不把完整 retrieved chunks 写入 trace？
12. 为什么 Agent Eval Dataset 只做 30 条？
13. 为什么单轮和多轮共用 turns schema？
14. Task Success 怎么算？
15. Planning Quality 为什么用 0/1/2？
16. 为什么 Planning 不做字符串完全匹配？
17. Trajectory Correctness 检查什么？
18. 为什么并行 Send 不适合完整 strict match？
19. required / forbidden tool 怎么评？
20. unnecessary Web 如何识别？
21. retry overuse 如何识别？
22. Efficiency 为什么不合成一个总分？
23. 一个答案正确但轨迹很差的 Agent 如何被识别？
24. Multi-turn Case 如何关联同一 trace？
25. Failure Case 如何定位回 trajectory.jsonl？
26. DeepSeek Judge 是否意味着“100% 本地”？
27. 当前 Agent Eval 的局限是什么？
```

每项包含：

```text
标准回答
继续追问
代码位置
```

---

# 53. 删除清单

新 Runtime 与 Local Harness 全部验证成功后 code search。

重点检查：

```text
LangSmith traceable import
LangSmith client / dataset / evaluate 调用
LANGSMITH_API_KEY / LANGSMITH_PROJECT 读取

SearchGraph.llm_db_search
旧 db_tool_node direct execution
旧 direct database_search runtime path
重复 Retrieval helper
```

若确认无有效引用：

```text
删除
```

不要保留：

```text
old_search_mode
legacy_search
langsmith_mode
compatibility flag
```

必须保留：

```text
SearchRequest
AgentPlannerPolicy
UnifiedRetrievalExecutor
DashScopeReranker
LocalTraceCollector
```

---

# 54. Stage 4 最终脚本链

最终只新增 / 保留：

```text
scripts/
├── 11_prepare_stage4_agent_eval.py
└── 12_eval_stage4_agent.py
```

不继续增加更多 Stage 4 脚本。

---

# 55. Definition of Done

- [ ] WSL + `conda activate rag`
- [ ] 正确目标仓库 `FELIN0366/medical-rag`
- [ ] 已审计 Codex 当前未提交改动，没有直接重置有用代码
- [ ] Stage 1 / Stage 2 参数未改
- [ ] SearchGraph Runtime 正式使用 `AgentPlannerPolicy`
- [ ] SearchGraph Runtime 正式使用 `UnifiedRetrievalExecutor`
- [ ] SearchGraph Runtime 正式使用 `qwen3.7-text-rerank`
- [ ] Final Top-5 进入现有 RAG Generate
- [ ] 旧 ToolNode direct DB Runtime 主链删除
- [ ] Retrieval provenance metadata 补齐
- [ ] 删除 Stage 4 中全部 LangSmith / Langfuse tracing 依赖
- [ ] `LocalTraceCollector` 完成
- [ ] Local Trace 不发任何 observability 网络请求
- [ ] `MedicalAgent / SearchGraph` 核心 trajectory 可在本地 JSONL 重建
- [ ] Multi-turn 可通过同一 `trace_id / session_id` 聚合
- [ ] 并行 subquery 可通过 `subquery_id` 区分
- [ ] Agent Behavior Eval Dataset = 30
- [ ] Simple single-hop = 5
- [ ] Rewrite = 5
- [ ] Decompose = 5
- [ ] Clarification = 4
- [ ] KB sufficient = 4
- [ ] KB insufficient = 3
- [ ] Multi-turn = 4
- [ ] Task Success evaluator
- [ ] Planning Quality evaluator
- [ ] Trajectory / Tool Correctness evaluator
- [ ] Efficiency evaluator
- [ ] 30 条正式 Agent Eval 完成
- [ ] `trajectory.jsonl` 留档
- [ ] `run_summaries.jsonl` 留档
- [ ] Query-level predictions 留档
- [ ] Evaluator results 留档
- [ ] Failure cases 留档
- [ ] `artifacts/stage4/*` 生成
- [ ] 两份 Stage 4 文档生成
- [ ] 未实现 LangSmith / Langfuse 平台
- [ ] 未实现 Evidence Grade
- [ ] 未实现 Citation Verification
- [ ] 未实现 Safety Benchmark
- [ ] 未实现 Memory Benchmark
- [ ] 未新增 Agent 功能 Stage

完成后停止。

---

# 56. 最终项目结构

```text
Stage 1
Multi-source Corpus
+ Ingestion
+ Chunking
+ Milvus Retrieval Foundation

↓

Stage 2
Hybrid Retrieval
+ Learned Reranker
+ Dynamic Retrieval Planner
+ Retrieval Evaluation

↓

Stage 4
Stage 2 Runtime Integration
+ Local Trajectory Harness
+ Agent Behavior Dataset
+ Agent Evaluation
```

最终评测分工：

```text
                    Evaluation
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      Retrieval Evaluation     Agent Evaluation
          Stage 2                 Stage 4
             │                     │
        Recall / MRR          Task Success
        nDCG                  Planning
        Reranker Gain         Trajectory / Tool
        Route Cost            Efficiency
```

最终面试定位：

> Stage 2 证明 Retrieval / Ranking 是否有效；Stage 4 在 LangGraph execution 层构建轻量级本地 Agent Evaluation Harness，证明 Agent 的 Planning、Tool Routing、Trajectory 与执行效率是否合理，同时避免把完整医疗 trajectory 额外上传第三方 observability 平台。

完成 Stage 4 后，不再扩展项目功能。
