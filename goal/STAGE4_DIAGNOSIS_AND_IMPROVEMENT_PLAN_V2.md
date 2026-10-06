# Stage 4 Diagnosis & Improvement Plan v2
# Original Graph Topology Diff → Legacy Cleanup → Targeted Regression

> **目标仓库**：`FELIN0366/medical-rag`  
> **只读原始参考**：`yolo-hyl/medical-rag`  
> **原始基线 Commit**：`4ec5871ad874344d228b19aeed7ab38c4ba0b940`  
> **当前诊断 Baseline**：`artifacts/stage4_diagnosis/`  
> **当前正式结果**：
> - 30/30 cases
> - 555 trajectory events
> - Task Success = 53.33%
> - Trajectory Correct = 46.67%
> - Planning Quality = 1.00 / 2
> - P50 = 10.59s
> - P95 = 83.04s
> - 无 execution exception / unclosed span / case timeout
>
> **本轮原则**：
>
> 1. 不先凭结果猜 Prompt；
> 2. 先把当前 LangGraph 节点、边、状态流与原始项目逐项对齐；
> 3. 区分“Stage 4 新引入回归”与“原始架构本身的设计局限”；
> 4. 完全清除旧 Database ToolNode / direct DB execution 主链；
> 5. 只在完成 topology audit 后做最小定向修复；
> 6. Stage 1 / Stage 2 Retrieval 参数全部冻结。

---

# 1. 为什么先做 Original Topology Diff

当前 Stage 4 同时修改了：

```text
MedicalAgent tracing wrapper
SearchGraph local retrieval node
Stage 2 online service
Agent Eval Runner
Local trajectory
Web timeout / fallback
```

因此出现行为失败时，不能直接认为：

```text
Prompt 不好
```

也不能直接认为：

```text
LangGraph 被改坏
```

正确顺序：

```text
Original Runtime Graph
        │
        ▼
Current Runtime Graph
        │
        ▼
Semantic Topology Diff
        │
        ├── 不应变化的地方发生变化
        │      → 先恢复
        │
        └── Topology 一致
               → 再看 Runner / State / Prompt / Router
```

---

# 2. Original MedicalAgent Graph——冻结参考

只读：

```text
yolo-hyl/medical-rag
src/MedicalRag/agent/MedicalAgent.py
```

对应本仓库历史：

```bash
git show 4ec5871ad874344d228b19aeed7ab38c4ba0b940:src/MedicalRag/agent/MedicalAgent.py
```

原始语义拓扑：

```text
                     START
                       │
                 route_entry
             ┌─────────┴─────────┐
             │                   │
   no background_info      has background_info
             │                   │
            ask         check_update_background
             │                   │
      route_ask_again             │
       ┌─────┴─────┐              │
       │           │              │
  need ask       pass             │
       │           │              │
      END   extract_ask_and_reply │
                   └───────┬──────┘
                           ▼
                      split_query
                           │
                      Send(search_one)
                    ┌──────┼──────┐
                    ▼      ▼      ▼
                   Q1     Q2     Q3
                    └──────┼──────┘
                           ▼
                         answer
                           │
                          END
```

关键状态语义：

```text
need_ask=True
→ 当前 graph invocation 结束
→ 等用户下一次 answer(user_input)

need_ask=False
→ extract_background
→ split / rewrite
→ retrieval
```

`MedicalAgent.answer()`：

```text
每次新用户 turn：
- 设置 curr_input
- reset sub_query_results
- 保留 asking_messages / background_info / dialogue_messages / summaries
- 再 invoke 同一 graph
```

---

# 3. Current MedicalAgent Graph 对比结论

当前节点改名：

```text
ask                    → clarification
extract_ask_and_reply  → background_update
answer                 → gather_answer
```

但当前连接：

```text
START
→ clarification / check_update_background

clarification
→ END / background_update

background_update
→ split_query

check_update_background
→ split_query

split_query
→ Send(search_one)

search_one
→ gather_answer

gather_answer
→ END
```

与原始项目**语义同构**。

因此：

> **当前 multi-turn 失败不能首先归因于 MedicalAgent 的 LangGraph edge 被 Stage 4 改坏。**

当前额外差异主要是：

```text
LocalTrace wrapper
turn_index
SearchGraph metadata
gather_answer 回传 state 时移除 sub_query_results
```

其中：

```text
gather_answer 不回传 sub_query_results
```

是为了避免 `Annotated[..., add]` reducer 再次追加聚合结果，应保留，不要为了“恢复原版”回退。

---

# 4. Original SearchGraph——冻结参考

只读：

```text
yolo-hyl/medical-rag
src/MedicalRag/agent/SearchGraph.py
```

原始拓扑：

```text
db_search
    │
    ▼
web_search（若 enabled）
    │
    ▼
rag
    │
    ▼
judge
 ┌──┼────┐
 ▼  ▼    ▼
pass retry fail
 │    │    │
 ▼    └→ rag
finish_success
      │
     END

fail → finish_fail → END
```

Fast mode：

```text
db_search / web_search
→ rag
→ END
```

---

# 5. Current SearchGraph 对比结论

当前：

```text
retrieve
→ web_search
→ rag
→ judge
→ pass / retry / fail
```

与原始拓扑**完全保持同一个后半段 Workflow**。

唯一算法替换：

```text
Original:
db_search
→ LLM database_search Tool Call
→ ToolNode
→ KnowledgeBase.search()

Current:
retrieve
→ OnlineRetrievalService
→ AgentPlannerPolicy
→ UnifiedRetrievalExecutor
→ Reranker
```

所以：

> **SearchGraph 的 topology 没有被重新设计；只是把第一个 local retrieval node 替换成 Stage 2 已验证主链。**

这说明当前主要行为失败应继续检查：

```text
Eval Runner semantics
Clarification / Rewrite decision boundary
Decomposition decision
Web Router
```

而不是重写整个 SearchGraph。

---

# 6. P0：完全删除旧 Database ToolNode / Direct DB Execution

当前旧执行逻辑仍残留在：

```text
src/MedicalRag/agent/tools/AgentTools.py

make_database_search_tool()
→ get_kb(...).search(...)
```

同时：

```text
AgentPlannerPolicy
```

只是为了复用 Tool Schema 而创建这个**可执行 database_search Tool**。

这与当前最终架构不一致。

## 6.1 最终目标

彻底不存在：

```text
llm_db_search
db_search_node
db_tool_node
db_search_llm
make_database_search_tool
database_search() direct executor
```

以及仅服务这些路径的：

```text
get_kb import
SearchRequest import
MedicalHybridKnowledgeBase import
DB ToolNode compatibility helper
```

如果无其他有效引用，全部删除。

---

# 7. Planner 不再依赖“可执行 Database Tool”

当前正确职责：

```text
Planner
→ 只选择 S / T / B

Executor
→ 唯一真正访问 Milvus
```

因此新建纯规划 Schema，例如：

```python
class RetrievalPlannerDecision(BaseModel):
    selected_channels: list[AnnsField]
```

允许：

```text
summary_dense
text_dense
text_sparse
```

至少一路。

Planner 使用：

```text
structured output
```

或：

```text
schema-only tool binding
```

获得 `RetrievalPlannerDecision`。

不得再构造一个内部能够执行：

```text
get_kb().search()
```

的 `database_search` Tool。

最终：

```text
AgentPlannerPolicy
     │
     ▼
RetrievalPlannerDecision
     │
     ▼
normalize_plan()
     │
     ▼
SearchRequest
     │
     ▼
UnifiedRetrievalExecutor
```

---

# 8. AgentTools 最终职责

清理后：

```text
AgentTools
├── web_search
└── calculator（若仍实际使用）
```

如果 calculator 无实际引用，也 code search 后删除。

`AgentTools` 不再拥有：

```text
database_search
```

SearchGraph 中唯一 `ToolNode` 允许是：

```text
network_tool_node
```

用于 Web Search。

不要因为用户要求删除 Database ToolNode，而误删仍在正式 Web Runtime 中使用的 Web ToolNode。

---

# 9. Legacy Cleanup 验收

必须执行：

```bash
grep -R "llm_db_search" -n src tests
grep -R "db_tool_node" -n src tests
grep -R "db_search_llm" -n src tests
grep -R "make_database_search_tool" -n src tests
grep -R "\"database_search\"" -n src tests
```

目标：

```text
旧 Runtime Executor 引用 = 0
```

如果 evaluator 的字符串：

```text
required_tools=["database_search"]
```

仍作为**行为抽象标签**存在，可以保留。

但必须在文档说明：

> `database_search` 在 Eval Schema 中表示“发生一次本地 Retrieval Pipeline”，不再对应一个 LangChain database ToolNode。

---

# 10. Multi-turn 的第一诊断优先级：Eval Runner，而不是 Graph Edge

当前 Runner：

```python
for turn in case.turns:
    state = agent.answer(turn)
    ...
    if ask_obj and ask_obj.need_ask:
        break
```

这与原始 Agent 多轮语义不一致。

原始 Agent 中：

```text
need_ask=True
→ 只结束当前 graph invocation
→ 等待下一次用户输入
```

而当前 Eval Runner 将它解释成：

```text
need_ask=True
→ 结束整个 Eval Case
```

对于：

```text
agent-27 ~ agent-30
```

这会直接截断冻结的后续 turns。

---

# 11. Multi-turn Runner 修复

对于：

```text
primary_category == multi_turn_context
```

必须：

```text
顺序执行完整 turns[]
```

即使前一 turn 返回 clarification：

```text
Turn 1
→ Agent clarification

Turn 2
→ 仍然作为用户下一轮输入继续 agent.answer()

Turn 3
→ 继续
```

这才模拟原始 `MedicalAgent.answer()` 的真实多轮使用。

只有：

```text
primary_category == clarification_needed
AND
expectation.terminal_behavior == clarification
```

才允许在首次符合预期 clarification 时结束该 Case。

---

# 12. 不要错误 reset 原始多轮 State

对齐原版时要保留：

```text
asking_messages
background_info
dialogue_messages
multi_summary
running_summary
curr_ask_num
```

跨 turn 演化。

每 turn 只保留原本应该 reset 的：

```text
curr_input
sub_query_results
```

不要为了 Eval 方便每 turn new 一个 MedicalAgent。

同一 Case：

```text
一个 MedicalAgent instance
一个 session_id
多个 answer(turn)
```

---

# 13. Rewrite Failure：不是 Stage 4 Graph Regression

原始 MedicalAgent 本来就是：

```text
Clarification
先于
Rewrite / Decompose
```

因此当前：

```text
rewrite_needed 0/5
```

不能简单归因于 Stage 4 改坏图。

这里是**原始设计边界与新 benchmark 的冲突**：

```text
ask_judge
把“口语模糊”
解释成
“医学信息不足”

→ rewrite node 永远到不了
```

Topology Audit 完成后，再定向修改 Clarification vs Rewrite 的决策边界。

不要调整 edge 顺序作为第一选择。

---

# 14. Rewrite Eval Case 先做 Label Audit

以下样本单轮没有真实 antecedent：

```text
agent-07:
“那个压上去以后人会有啥感觉？”

agent-08:
“上回说的盐，到底一天别超过多少？”
```

此时 Agent 追问不是显然错误。

因此先修 Eval：

```text
agent-07:
“血压老往上蹿，人一般会有啥感觉？”

agent-08:
“高血压平时盐到底一天别超过多少？”
```

仍测试：

```text
口语化
→ Rewrite
→ 不拆分
```

而不是测试缺失上下文。

---

# 15. Clarification vs Rewrite Prompt 定向修复

只有在 topology / label audit 完成后修改 Prompt。

明确职责：

```text
Clarification:
缺少会改变风险、安全、诊断/处理决策的事实

Rewrite:
表达口语化
术语不标准
省略但能从当前 turn / 已有 history 恢复
检索表达不自洽
```

禁止：

```text
只因为“表达不专业”
→ clarification
```

---

# 16. Decomposition Diagnosis

原始和当前 `split_query` edge 一致。

因此：

```text
missing decomposition
```

首先看：

```text
HANDLE_QUERY_SYSTEM_PROMPT
+
实际 planner output
```

而不是重连 graph。

补充通用 multi-intent 规则：

```text
如果一个请求包含 >=2 个可独立检索、独立回答的 Evidence Need
→ need_split=True
```

例如：

```text
症状 + 饮食
分级 + 风险因素
诊断阈值 + 食盐
```

不要只围绕：

```text
诊断 vs 治疗
鉴别诊断
并发症
```

---

# 17. Web Router Diagnosis

Original / Current topology：

```text
local retrieval
→ web_router
→ optional web
→ RAG
```

没有变化。

所以：

```text
4 unnecessary web
1 missing web
```

不是 edge regression。

优先检查：

```text
A. 新 Top-5 Reranked Evidence 改变了 Router 输入
B. 原始 WEB_SEARCH_JUDGE prompt 过宽
C. Router timeout / parse fallback 是否导致 false negative
```

Prompt 应收紧为：

```text
实时/最新信息 → Web required
Local Evidence 已足够 + 非时效问题 → Web forbidden
```

但先从 trajectory 逐 Case 验证再改。

---

# 18. agent-25 Label 修正

当前：

```text
“我所在城市今天的空气质量……”
```

没有城市。

仅 Web Search 无法解决。

推荐修改为：

```text
“今天广州的空气质量是否适合高血压患者户外运动？”
```

保持：

```text
web_policy=required
```

不改变 category 数量。

---

# 19. Current vs Original 必须自动生成 Diff Report

Codex 必须生成：

```text
artifacts/stage4_regression_v2/original_graph_diff.md
```

至少包含：

## MedicalAgent

```text
Original node
Current node
Original outgoing edges
Current outgoing edges
semantic equivalent?
reason
```

## SearchGraph

同样表格。

## State Lifecycle

比较：

```text
ask_obj
curr_ask_num
asking_messages
background_info
dialogue_messages
sub_query_results
multi_summary
running_summary
```

跨 turn 生命周期。

---

# 20. 添加 Graph Invariant Tests

不要只靠人工阅读。

新增测试，确保当前最终拓扑符合我们冻结语义：

## MedicalAgent

```text
START
→ clarification | check_update_background

clarification
→ END | background_update

background_update
→ split_query

check_update_background
→ split_query

split_query
→ Send(search_one)

search_one
→ gather_answer

gather_answer
→ END
```

## SearchGraph

```text
retrieve
→ web_search | rag

web_search
→ rag

rag
→ judge | END

judge
→ finish_success | rag | finish_fail
```

不要要求节点名与原项目完全相同，只检查 semantic topology。

---

# 21. Trace 完整性

在 topology 修复后再补：

```text
subquery_id
candidate_pks
final_pks
```

特别是并行 `Send`：

```text
Q1 / Q2
```

必须能区分各自：

```text
retrieval_planner
retrieval_executor
reranker
web_router
judge
```

不能只靠时间顺序猜。

---

# 22. Latency Diagnosis

在结构问题解决后，用 trajectory span 输出：

```text
latency_by_node.json
```

节点：

```text
clarification
background_update
split_query
retrieval_planner
retrieval_executor
reranker
web_judge
web_tool_planner
tavily_search
rag_generate
judge
gather_answer
```

统计：

```text
count
mean
p50
p95
max
```

再决定 P95=83s 来自哪里。

---

# 23. Regression 顺序

不要一次改完所有东西再跑。

建议：

```text
Step 1
Original Graph Diff
+ Legacy DB Tool cleanup
+ Graph invariant tests

Step 2
Fix multi-turn Eval Runner
→ 只跑 agent-27~30

Step 3
Audit/fix rewrite labels
+ Clarification/Rewrite boundary
→ 跑 agent-06~10 + clarification 16~19

Step 4
Decomposition prompt
→ 跑 agent-11~15

Step 5
Web Router
→ 跑 agent-20~26

Step 6
完整 30-case regression
```

每一步必须保留局部结果，不允许为了最终分数跳过失败归因。

---

# 24. 不允许修改的部分

```text
Stage 1 corpus
Parser / Chunking
Milvus schema
Embedding model
Stage 2 Top-50
RRF k=60
Candidate Top-30
Final Top-5
qwen3.7-text-rerank
```

本轮不是 Retrieval 调优。

---

# 25. Final Regression

保留：

```text
artifacts/stage4_diagnosis/
```

作为不可覆盖 baseline。

新结果：

```text
artifacts/stage4_regression_v2/
```

必须包含：

```text
original_graph_diff.md
metrics.json
efficiency.json
failure_cases.jsonl
predictions.jsonl
evaluator_results.jsonl
trajectories.jsonl
latency_by_node.json
before_after.json
```

---

# 26. Definition of Done

- [ ] 完成 `yolo-hyl/medical-rag@4ec5871` 与当前 MedicalAgent topology 对比
- [ ] 完成原始与当前 SearchGraph topology 对比
- [ ] 证明或定位实际 topology regression
- [ ] 不盲目回滚正确的新逻辑
- [ ] `make_database_search_tool` 删除
- [ ] executable `database_search` Tool 删除
- [ ] DB ToolNode / db_search_llm / llm_db_search 全清理
- [ ] Planner 改用纯 Schema Decision
- [ ] SearchGraph 唯一 local executor 为 OnlineRetrievalService
- [ ] Multi-turn Runner 与原始 `answer()` 语义一致
- [ ] 4 条 multi-turn case 全量按 turns 执行
- [ ] Rewrite Dataset 标签审计完成
- [ ] Clarification / Rewrite boundary 修复
- [ ] Decomposition 通用规则修复
- [ ] Web Router 定向修复
- [ ] agent-25 修正
- [ ] Graph invariant tests
- [ ] subquery_id trace
- [ ] candidate/final PK trace
- [ ] latency_by_node
- [ ] Stage 1 / 2 参数未改变
- [ ] 30-case regression 完成
- [ ] 更新 Stage 4 文档
- [ ] 完成后停止扩功能
