# Stage 4：本地 Agent 行为评测面试问答

## 1. 为什么 Retrieval Eval 后还需要 Agent Eval？

标准回答：检索评测验证候选和排序，Agent 评测验证追问、改写、拆解、工具和 Web 决策是否正确。继续追问：答案正确但轨迹差如何发现？代码位置：`agent_evaluation/evaluators.py`。

## 2. Stage 2 与 Stage 4 分别评价什么？

标准回答：Stage 2 评价 Recall、MRR、nDCG 与重排/选路；Stage 4 评价任务、规划、轨迹和效率。继续追问：为什么不重复 MRR？代码位置：`evaluation/retrieval_runner.py`、`agent_evaluation/runner.py`。

## 3. 为什么不继续增加 Evidence Grade 或 Citation？

标准回答：它们是独立目标，会扩大标签与判定边界，当前只冻结 Agent 行为。继续追问：以后如何接入？代码位置：`goal/STAGE4_AGENT_EVALUATION_IMPLEMENTATION_SPEC_V1.md`。

## 4. MedicalAgent 与 SearchGraph 的关系是什么？

标准回答：前者管理多轮和子查询，后者处理一个子查询的检索、Web、生成与 Judge。继续追问：并行在哪里发生？代码位置：`agent/MedicalAgent.py`。

## 5. Stage 2 Stack 如何接入真实 SearchGraph？

标准回答：`retrieve` 节点调用薄服务层，服务层依次调用 Planner、Executor、Reranker。继续追问：是否复制了 RRF？代码位置：`retrieval/service.py`。

## 6. 为什么必须共用在线主链？

标准回答：评测与生产若用不同检索算法，评测结论无法解释线上行为。继续追问：旧链在哪里删除？代码位置：`agent/SearchGraph.py`。

## 7. 为什么用 LocalTraceCollector？

标准回答：本阶段只需本地、可复现的行为轨迹，不把数据发送至观测云端。继续追问：生产 API 会落盘吗？代码位置：`agent_evaluation/local_trace.py`。

## 8. 为什么只记录核心节点？

标准回答：节点必须支持决策归因，细粒度 helper 会制造噪声且增加数据暴露面。继续追问：记录哪些节点？代码位置：`agent/SearchGraph.py`。

## 9. 为什么只做 30 条？

标准回答：它是人工可核对的行为回归集，不冒充统计显著性的医学基准。继续追问：类别怎么控制？代码位置：`agent_evaluation/dataset.py`。

## 10. 为什么单轮与多轮共用 turns？

标准回答：Runner 可以统一执行，且多轮的 session 关联得到明确验证。继续追问：如何确保同 session？代码位置：`agent_evaluation/runner.py`。

## 11. Task Success 怎么算？

标准回答：Judge 根据冻结 requirement 和最终输出给 0/1。继续追问：Judge 读 chunk 吗？代码位置：`agent_evaluation/evaluators.py`。

## 12. 为什么 Planning Quality 是 0/1/2？

标准回答：规划质量有可接受的中间态，二元评分会丢失冗余与轻微遗漏。继续追问：N/A 如何处理？代码位置：`agent_evaluation/evaluators.py`。

## 13. 为什么不做字符串完全匹配？

标准回答：等价改写不应被误判，重点是意图覆盖和互补性。继续追问：如何保持一致？代码位置：`AgentExpectation.expected_topics`。

## 14. Trajectory Correctness 检查什么？

标准回答：终态、拆解范围、工具、Web 与重试上限。继续追问：为什么不是完整 action 序列？代码位置：`trajectory_tool_correctness`。

## 15. 为什么 Send 不适合 strict match？

标准回答：并行子任务显示顺序可变，顺序不是行为正确性的必要条件。继续追问：保留哪些约束？代码位置：`agent/MedicalAgent.py`。

## 16. required / forbidden tool 怎么评？

标准回答：由本地 Retrieval Pipeline 与 Web 实际调用计数导出行为集合，再做集合约束。`database_search` 仅是“发生本地 Retrieval Pipeline”的评测标签；Planner 的 `RetrievalPlannerDecision` 是纯通道选择 Schema，不是执行 Tool。继续追问：代码位置：`retrieval/policy.py`、`agent_evaluation/evaluators.py`。

## 17. unnecessary Web 如何识别？

标准回答：冻结 `web_policy=forbidden` 且实际 Web 调用大于零。继续追问：Web required 如何判？代码位置：`trajectory_tool_correctness`。

## 18. retry overuse 如何识别？

标准回答：比较累计 Judge retry 与 expectation 上限。继续追问：没有上限怎么办？代码位置：`AgentExpectation.max_retry_count`。

## 19. 为什么 Efficiency 不合成总分？

标准回答：调用、延迟和 token 是不同量纲，单一加权分会掩盖权衡。继续追问：报告哪些指标？代码位置：`aggregate_efficiency`。

## 20. Local trajectory 与 Stage 2 Harness 如何分工？

标准回答：前者解释 Agent 行为，后者评测检索质量；两者共享线上 Retrieval Stack。继续追问：谁保存 Gold？代码位置：`evaluation/`、`agent_evaluation/`。

## 21. 答案正确但轨迹很差如何识别？

标准回答：Task Success 可以通过，而 Trajectory 或 Efficiency 会暴露无谓拆解、Web 或重试。继续追问：失败如何归类？代码位置：`runner.py`。

## 22. Multi-turn 如何关联？

标准回答：同 Case 的 turns 使用同一个 Agent 实例和 `stage4-<case_id>` session。继续追问：轨迹里如何查看？代码位置：`trajectories.jsonl`。

## 23. Failure Case 如何回看？

标准回答：failure artifact 保存 expectation、实际规划、工具、答案和 trajectory_id，可按 session 筛 JSONL。继续追问：是否含 chunk？代码位置：`LocalTraceCollector`。

## 24. 当前局限是什么？

标准回答：30 条不是医疗临床正确性基准，Judge 有模型偏差，token usage 可能不可用，实时 Web 依赖既有服务。独立 holdout 的 Task Success 仍为 53.33%，不能把诊断集上的局部改善宣传为泛化提升。继续追问：哪些没有实现？代码位置：`docs/stage4_agent_evaluation_implementation.md`。
