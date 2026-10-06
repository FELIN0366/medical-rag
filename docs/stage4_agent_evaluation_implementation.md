# Stage 4：本地 Agent 行为评测实现说明

## 1. 完整 Pipeline

`MedicalAgent` 负责追问、背景更新、改写、拆解与汇总；每个子查询进入 `SearchGraph`。`SearchGraph` 的唯一本地检索节点为 `retrieve`，顺序调用 Stage 2 的 Planner、统一执行器和重排器，最终 Top-5 文档进入既有 Web Router、RAG Generate 与 Judge 重试回路。

## 2. Stage 2 Retrieval Stack 接管 SearchGraph

`retrieval/service.py` 的 `OnlineRetrievalService` 只编排 `AgentPlannerPolicy`、`UnifiedRetrievalExecutor` 与 `DashScopeReranker`，不复制向量编码、Milvus 查询、RRF、Planner Prompt 或 Reranker 请求逻辑。Stage 1/2 的 Top-50、RRF(60)、Candidate Top-30、Final Top-5 参数保持不变。

## 3. 删除的旧 Runtime 路径与纯规划决策

`SearchGraph` 已删除 `llm_db_search`、`db_search_tool`、`db_tool_node` 与 `db_search_llm`；`AgentTools` 也不再提供可执行的 `database_search` Tool。`AgentPlannerPolicy` 改用纯 Pydantic Schema `RetrievalPlannerDecision(selected_channels)`，只选择 S/T/B 通道。唯一访问 Milvus 的在线路径是 `UnifiedRetrievalExecutor`。

评测 Schema 中的 `required_tools=["database_search"]` 仅是行为标签，表示发生了一次本地 Retrieval Pipeline，不再对应 LangChain Tool 或 ToolNode。

## 4. Provenance

Online `SearchRequest` 显式请求 `pk`、三种文本表示、来源、文档/切片标识、科室、标题、章节路径与页码。`retrieval_info` 同时保存通道、Planner fallback、候选数及 Planner/Retrieval/Reranker 延迟。

## 5. 本地轨迹设计

本阶段不接入云端观测平台。`LocalTraceCollector` 仅在 Eval Runner 注入，收集 `clarification`、`background_update`、`split_query`、`search_one`、`retrieval_planner`、`retrieval_executor`、`reranker`、`web_router`、`web_search`、`rag_generate`、`judge` 与 `gather_answer` 的行为元数据。普通 API Runtime 不创建收集器，也不持久化真实用户轨迹。

## 6. JSONL Trajectory

`trajectories.jsonl` 按 event 保存 case、session、turn、`subquery_id`、事件名和受限元数据；收集器剔除 `documents`，并截断字符串。每个子查询的检索事件同时记录 `candidate_pks` 与 `final_pks`，可区分并行 `Send` 分支而无需落盘 chunk 正文。它用于轨迹规则、效率统计和失败回看，不作为 DeepSeek Judge 输入。

## 7. Agent Eval Dataset

初始诊断集为 `data/eval/agent_behavior_cases.jsonl`。在根据该集定位问题后，最终独立评测改用未参与后续调整的 holdout：`data/eval/agent_behavior_cases_holdout_v2.jsonl`，冻结副本位于 `artifacts/stage4_holdout_v2/agent_eval_dataset.jsonl`，其 SHA-256 为 `3eb59238dbce52fbc2703fab0e3a8c44aa05fae08d3fb6547caab0c1e6bf3f6e`。

## 8. Dataset 分布

30 条用例固定为：simple_single_hop 5、rewrite_needed 5、decompose_multi_intent 5、clarification_needed 4、kb_sufficient_no_web 4、kb_insufficient_web 3、multi_turn_context 4。多轮用例统一使用 `turns`，同一 Case 固定复用一个 Agent session。

## 9. Dataset Schema

`AgentEvalCase` 含 `case_id`、唯一主类别、非空 `turns` 与 `AgentExpectation`。Expectation 固定终态、改写/拆解、子查询上下限、主题、required/forbidden tools、Web policy、重试上限和答案要求；它不是检索 Gold，也不进入 Milvus。

## 10. Task Success

Task Success 使用低温 DeepSeek Judge，输入仅为冻结 turns、终态/答案要求和 Agent 最终输出。结果为 0/1 与简短 reason。追问用例判断追问是否针对信息缺口。

## 11. Planning Quality

Planning Judge 只读取 turns、冻结的 rewrite/decompose expectation、实际改写和子查询，输出 0/1/2。它评价必要性、覆盖、冗余和语义保真，不做子查询字符串完全匹配。未进入规划的追问用例记为 N/A，排除出均值。

## 12. Trajectory / Tool Correctness

确定性评测检查终态、子查询范围、required/forbidden tools、Web policy 与重试上限。它允许 `Send` 并行子任务顺序不同，只检查关键约束，输出总结果与各规则布尔值。

## 13. Efficiency

Efficiency 不调用 Judge，不合成单一分数。它保存 turns、子查询、route、数据库/Web 调用、重试、工具调用、延迟及真实 token usage。当前 Provider 没有稳定可用 token usage 时写 `null`，不估算伪造。

## 14. Artifacts

正式 Runner 生成数据集副本、报告、predictions、evaluator results、metrics、efficiency、failure cases、`trajectories.jsonl` 和脱敏 config snapshot。回归汇总脚本还生成 `original_graph_diff.md`、`latency_by_node.json` 和 `before_after.json`。Stage 4 不生成或复述 Recall、MRR、nDCG、Reranker Rescue/Damage；这些仍以 Stage 2 artifacts 为准。

## 15. 已验证状态

已通过 holdout 数据集冻结校验：30 条、类别分布正确。图不变量、纯 Planner Schema、并行分支失败隔离、Runner 多轮语义、轨迹与 artifact 测试共 33 项通过。本文不把这些测试伪称为真实在线 Agent Eval。

## 16. 独立 Holdout 正式结果

`artifacts/stage4_holdout_v2/` 中已完成 30/30 条真实在线评测：Task Success 为 53.33%，Trajectory Correct 为 60.00%，Planning Quality 均值为 1.39/2。相对诊断基线，Trajectory Correct 提升 13.33 个百分点，Task Success 未提升；这说明旧诊断集的局部改善不能被表述为泛化性能提升。

节点延迟显示 P50 为 37.56 秒、P95 为 100.90 秒。高耗时节点主要为 Tavily 搜索、RAG Generate、Web Judge 与 Answer Judge，详见 `latency_by_node.json`。实时 Web 的提供方超时仍会出现，但会按受控错误回退，且单个并行子查询超时不会中止同一 Case 的其余成功分支。

## 17. 运行前置与限制

真实 smoke 与 30 条正式 Eval 需当前 shell 具备 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`DASHSCOPE_WORKSPACE_ID`，并依赖既有 Milvus 与 Web Search 配置。缺少任一变量时 `scripts/12_eval_stage4_agent.py` 报 `BLOCKED`，不会写伪造结果。DeepSeek Judge 从不接收完整轨迹、检索 chunk 或真实 API 用户历史。
