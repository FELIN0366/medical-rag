# Stage 2 面试问答

## 1. Retriever、RRF、Reranker、Planner 分别解决什么问题？

标准回答：Retriever 负责广召回，RRF 融合多种召回信号，Reranker 负责在固定候选池内做 query-document 精排，Planner 只决定该用哪些召回信号。继续追问：为什么不让 Reranker 直接搜索全库？代码位置：`retrieval/policy.py`、`retrieval/executor.py`、`retrieval/reranker.py`。

## 2. 为什么 Fixed Hybrid 是强 Baseline？

标准回答：它同时覆盖主题语义、正文证据与词面精确匹配，并固定 S+T+B、Top-50、RRF(60)，可隔离 Reranker 与 Planner 的增益。继续追问：单一路径为何不够？代码位置：`retrieval/policy.py`。

## 3. 为什么固定 Top-50 → Top-30 → Top-5？

标准回答：三路各取足够召回深度，RRF 形成可控的 30 条重排池，最终 5 条利于下游使用与人工检查。继续追问：为什么不是把全部文档送 Reranker？代码位置：`config/models.py`。

## 4. 为什么 Reranker 救不回 Candidate Pool 外的 Gold？

标准回答：重排只重新排列输入候选，不能检索未输入的文档。继续追问：如何定位是召回问题还是排序问题？代码位置：`evaluation/retrieval_runner.py`。

## 5. qwen3.7-text-rerank 为什么适合本项目？

标准回答：它支持 query-document 相关性重排，可用医疗“直接证据优先”的 instruct 约束排序目标，并通过 Model Studio 统一接入。继续追问：是否证明其优于所有模型？代码位置：`retrieval/reranker.py`。

## 6. 为什么使用 instruct？

标准回答：通用相关性容易把主题相近误排为证据相关；instruct 明确优先能直接回答的问题证据。继续追问：instruct 是否会替代人工标签？代码位置：`config/app_config.yaml`。

## 7. 为什么 relevance_score 不做全局 threshold？

标准回答：分数只在同一 query 的候选列表内排序，跨 query 的绝对分值不稳定；本阶段固定取 Top-5。继续追问：何时才需要阈值？代码位置：`retrieval/reranker.py`。

## 8. 为什么 Exp B-A 可以解释 Reranker Gain？

标准回答：B 完全复用 A 的 Candidate Top-30，唯一变化是重排。继续追问：怎样保证候选池未被重复检索改变？代码位置：`evaluation/retrieval_runner.py`。

## 9. 为什么 Exp C-B 可以解释 Planner Gain？

标准回答：两者使用同一 Reranker 和固定参数，差异只来自 Planner 选择的路由子集及其候选池。继续追问：如果 C 更差如何归因？代码位置：`evaluation/retrieval_runner.py`。

## 10. 为什么 Planner 本阶段只选择 Channel？

标准回答：把动作空间限制为 S/T/B 能让实验可解释，避免模型同时改变数据库调参。继续追问：后续是否能开放更多动作？代码位置：`retrieval/policy.py`。

## 11. 为什么不让 Planner 改 ef、metric、RRF k？

标准回答：这些参数会混淆选路收益、索引成本与召回变化，且增加不可复现实验自由度。继续追问：代码如何强制？代码位置：`retrieval/policy.py`。

## 12. Eval Mode 为什么预跑 S/T/B？

标准回答：它用于诊断每条 Gold 在哪一路存在，帮助分析 Planner Loss；这些结果不进入 Planner prompt。继续追问：是否泄漏评测信息？代码位置：`retrieval/executor.py`。

## 13. Online Mode 为什么只执行 Planner 选择 route？

标准回答：若后台全查就无法测量动态选路的实际成本，且会把“假选路”伪装为效率提升。继续追问：如何验证？代码位置：`evaluation/retrieval_runner.py`。

## 14. Planner Failure 如何归因？

标准回答：记录 selected channels、三条 raw route 的 Gold 名次和 fallback 错误；可判断是否漏选了有 Gold 的通道。继续追问：Planner 无 tool call 怎么办？代码位置：`retrieval/policy.py`。

## 15. 为什么需要 routes/query？

标准回答：质量相近时，更少的实际路由代表更低 retrieval work；只看 Recall 无法说明成本。继续追问：如何统计 1/2/3 route？代码位置：`evaluation/retrieval_runner.py`。

## 16. 为什么 QA 和 Literature 分开统计？

标准回答：QA 自检索和文献证据检索的数据分布、Gold 形态不同，混合平均会掩盖退化。继续追问：Cross-corpus 放哪里？代码位置：`evaluation/retrieval_metrics.py`。

## 17. partial relevant / hard negative 有什么价值？

标准回答：partial 让 nDCG 区分直接证据与部分相关，hard negative 用于检查主题相近但不能回答的问题。继续追问：Recall 为什么不把 partial 当 Gold？代码位置：`evaluation/retrieval_dataset.py`。

## 18. 为什么 Stage 2 不做 7-action 全量实验？

标准回答：当前目标是确认 Reranker 与可解释的选路收益；全排列会稀释样本和计算预算。继续追问：何时再扩展？代码位置：`docs/stage2_retrieval_reranker_planner_implementation.md`。

## 19. 为什么不做 Reranker model zoo？

标准回答：先固定一个可用模型，保证候选、标签、指标和选路比较稳定；模型横评是独立实验。继续追问：当前结果能否声明最佳模型？代码位置：`config/app_config.yaml`。

## 20. Stage 2 如何为 Evidence Grade / Citation 铺路？

标准回答：它已经保存 query、候选、最终排序、Gold 及相关性等级，可作为后续证据判断和引用绑定的输入；本阶段不生成答案或引用。继续追问：为什么现在不做？代码位置：`evaluation/retrieval_dataset.py`、`evaluation/retrieval_runner.py`。
