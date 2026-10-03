# Stage 2：检索、重排与选路实现说明

## 1. Stage 2 完整 Pipeline

Stage 2 只读取冻结的 `medical_knowledge` 集合：查询先用相同的 `qwen3.7-text-embedding-flash` 向量进入 `summary_dense`、`text_dense`，同时将原查询交给 Milvus 内置 BM25 的 `text_sparse`。固定基线使用三路 Top-50，经 RRF(k=60) 取 Candidate Top-30；Reranker 再从该候选池取 Final Top-5。Agent 实验只让 Planner 选择 S、T、B 的非空子集，参数始终由代码冻结。

本阶段不会创建、删除、重建或写入 Milvus collection，也不会进入答案生成、证据分级、引用、记忆或安全评估。

## 2. Reranker 配置与 API

配置位于 `src/MedicalRag/config/app_config.yaml` 的 `reranker`：模型为 `qwen3.7-text-rerank`，服务地址由 `DASHSCOPE_WORKSPACE_ID`、`cn-beijing` 拼接，鉴权读取 `DASHSCOPE_API_KEY`。每一个 query 对整个 Candidate Top-30 只发一次请求；Reranker 输入严格使用 `Document.page_content`，即冻结 chunk 的 `text`。请求中含有固定 medical-evidence `instruct`、`top_n` 和完整文献列表。客户端有 30 秒超时与最多 3 次重试，按返回的 `index` 和 `relevance_score` 排序，不把分数回写 Milvus。

## 3. Policy / Executor 解耦

`FixedHybridPolicy` 与 `AgentPlannerPolicy` 只产出规范化的 `RetrievalPlan`；`UnifiedRetrievalExecutor` 是唯一执行查询向量编码与 Milvus 检索的入口。`normalize_plan()` 丢弃 Planner 提供的任何 metric、ef、limit、RRF 和 source filter，只保留 S/T/B 的选择，因此数据库参数不可被模型改变。

## 4. Fixed Hybrid

固定策略为 S+T+B：dense 使用 COSINE、`ef=64`、Top-50；稀疏使用 BM25、`drop_ratio_search=0`、Top-50；多路使用 RRF(k=60)，最终 Candidate Top-30。它是所有增益比较的共同强基线。

## 5. Agent Planner

Agent 复用已有的 `database_search(search_config: SearchRequest)` Tool Schema、`call_db` 提示模板和工具绑定 LLM。提示只解释三条通道各自适合的问题，并要求选取最少但足够的通道。Planner 无 tool call 或解析失败时会回退到固定三路，并在 query 级 artifact 记录 `planner_error`；这不会被误记为成功的动态选路。

## 6. Reranker

Exp B 和 Exp C 使用同一 `DashScopeReranker` 实例类型、同一模型、同一 instruct 和同一 Top-30 → Top-5 规则。这样 B-A 的差异只来自重排，C-B 的差异只来自候选池和 Planner 选路。

## 7. Eval Dataset

`09_prepare_stage2_eval.py` 只读 collection：从 200 QA 中按 PK 排序、seed=42 固定抽取 30 条，并以每批 10 条调用 `deepseek-flash` 改写。若改写去标点后仍等于原题，会拒绝写入。20 条 Literature 标签由人工冻结，采用 direct=2、partial=1、hard negative=0。

Cross-corpus diagnostic 的标准版本只能接受人工核验的 5–10 条记录：原题应来自 QA，direct Gold 必须是能直接回答该 QA 的 literature chunk。`09_prepare_stage2_eval.py --cross-corpus-candidates-only` 会通过只读 BM25 写出人工核验候选；脚本不尝试从主题相似度自动伪造最终标签。

当前冻结语料缺少足够的不同 QA→文献 direct-evidence 配对。用户明确允许后，`--build-best-effort-cross` 会生成 5 条只含 partial 标签的诊断记录；这些记录的 `label_quality=best_effort`，不应被叙述为 direct-evidence Eval，也不进入 QA/Literature 主指标。

## 8. Metrics

QA 与 Literature 分开统计 Candidate Recall@30、Recall@5、MRR 和 graded nDCG@5。Recall/MRR 只将 direct evidence 当作 Gold；nDCG 使用 2/1/0 relevance。Cross-corpus 仅是诊断，不把其混入两个主集合的平均值解释。

## 9. Exp A/B/C

Exp A 保存 Fixed Hybrid Candidate Top-30 与 RRF Top-5；Exp B 完全复用 A 的候选池并重排；Exp C 让 Planner 在线选路，随后执行被选中的 route 并使用同一 Reranker。`10_eval_stage2_retrieval.py` 是三组实验唯一入口，避免产生多套不一致 runner。

## 10. Reranker Gain

`reranker_analysis.json` 保存 Rescue@5（A 未命中、B 命中）、Damage@5（A 命中、B 未命中）、Candidate Top-30 中 direct Gold 的平均最佳名次增益，以及 p50/p95 重排延迟。名次增益使用完整 reranked Top-30，不能用截断的 Top-5 代替。

## 11. Planner Gain

`planner_analysis.json` 记录平均 routes/query、1/2/3-route 比例、Planner 与 retrieval p50/p95 延迟以及 fallback 数量。只有 Agent 的质量接近 Fixed+Reranker 且实际 routes/query 更少，才可解释为在减少检索工作。

## 12. Route Cost

评测模式会预跑 S、T、B 三路 Top-50 并写入 `raw_routes.jsonl`，仅用于归因。线上 Agent 模式不会后台三路全查；它只执行 Planner 实际选择的非空子集。

## 13. Failure Analysis

`failure_cases.jsonl` 最多保存每类五条：Reranker Rescue、Reranker Damage、Planner Win、Planner Loss、BM25-only strong、Dense strong。Planner Loss 会额外记录 selected channels 和 Gold 在 summary/text/BM25 三路中的名次，以区分“漏选路由”与“所有 route 都未召回”。

## 14. API batching/cache

所有 Eval query 先按 20 条一批 embedding，写为 `query_embeddings.npz` 和 `query_embeddings_meta.json`。summary/text 配置一致时同一 query 只生成一次 dense vector，Fixed 与 Agent 复用同一缓存；BM25 仍使用原始 query。QA 改写也是每批 10 条，Rerank 则是每 query 一次完整候选列表请求。

## 15. 实际实验结果

已真实生成 30 条 QA Internal Eval（固定 seed=42、三批 DeepSeek 改写）和 20 条 Literature Eval；QA 改写与原题逐条不相同。DashScope Reranker endpoint smoke 返回有效 index 与数值 relevance score。完整 A/B/C 已在 2026-10-03 真实运行；55 条 query-level prediction 和三路 raw route 已保存。

| 数据集 | 实验 | Candidate Recall@30 | Recall@5 | MRR | nDCG@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| QA（30） | Fixed Hybrid | 1.0000 | 0.9667 | 0.9500 | 0.9544 |
| QA（30） | Fixed + Reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| QA（30） | Agent + Reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Literature（20） | Fixed Hybrid | 1.0000 | 1.0000 | 0.7583 | 0.7653 |
| Literature（20） | Fixed + Reranker | 1.0000 | 1.0000 | 0.9750 | 0.9339 |
| Literature（20） | Agent + Reranker | 1.0000 | 1.0000 | 0.9750 | 0.9339 |

Reranker 对 QA 的 Rescue@5 为 0.0333（1/30），Damage@5 为 0，平均最佳 Gold 名次增益为 0.3333；对 Literature 的 Rescue/Damage 均为 0，平均名次增益为 0.6500。QA Planner 平均选择 2.0333 条 route，Literature 为 2.0000；两者 fallback 均为 0。完整数值、延迟分位数与失败案例以 `artifacts/stage2/` 中的 JSON 为准。

当前也生成了用户允许的 5 条 `label_quality=best_effort` Cross-corpus 诊断记录；它们只含 partial 标签，不能视为 direct-evidence Gold，因而不进入上表主结论。

## 16. Known Limitations

当前 Stage 1 的 QA 与文献有主题重叠但直接可证据配对有限；Cross-corpus 必须人工审核。评测只度量检索，不说明医学答案正确性；Reranker 不能救回 Candidate Top-30 外的 Gold。文献来源中可能存在表格、格式噪声或内容时效性问题，这些应在后续人工标注与语料治理中处理，而不应在本阶段通过修改已冻结的语料掩盖。
