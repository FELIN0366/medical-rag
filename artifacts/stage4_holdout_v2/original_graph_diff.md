# 原始图与当前图的语义拓扑对比

冻结基线：`yolo-hyl/medical-rag@4ec5871ad874344d228b19aeed7ab38c4ba0b940`。本报告在生成时以 `git show` 读取了两个原始文件，并读取当前工作区文件。

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

- 通过：原始 MedicalAgent 包含 ask 节点
- 通过：原始 SearchGraph 包含 db_search 节点
- 通过：当前 MedicalAgent 包含 clarification 节点
- 通过：当前 SearchGraph 包含 retrieve 节点
- 通过：当前无旧数据库 ToolNode

结论：当前图未发生需要回退的拓扑回归。原始 `db_search` 的可执行 ToolNode 已被 `retrieve` 的 Stage 2 统一检索链替换；这是有意算法替换而非边断裂。
