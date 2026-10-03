# Stage 2 Implementation Spec v4
# Fixed Hybrid + Qwen Reranker + Structured Retrieval Planner + Key Retrieval Evaluation

> **目标仓库（唯一允许修改）**：`FELIN0366/medical-rag`  
> **参考仓库（只读）**：`WYW-1127/medicalrag`  
> **执行环境**：WSL Ubuntu  
> **Conda 环境**：`rag`  
> **Milvus**：Docker Standalone `milvusdb/milvus:v2.6.0`，`http://localhost:19530`  
> **Stage 1 Dense Embedding**：`qwen3.7-text-embedding-flash`，1024D，DashScope OpenAI-compatible API  
> **Stage 2 Reranker**：`qwen3.7-text-rerank`，Alibaba Cloud Model Studio / DashScope  
> **Planner**：复用现有 `SearchRequest + Tool Schema + Tool-bound LLM + deterministic Milvus executor`  
> **核心原则**：只保留一套 Retrieval Executor / Reranker / Evaluator；Fixed Hybrid 与 Agent Planner 只是两种 Retrieval Policy。  
> **阶段边界**：本阶段只完成 Fixed Hybrid、Reranker、Agent 动态选路和关键 Retrieval Evaluation；不进入 Evidence Grade、Citation、Safety、Persistent Memory、多模态或新的 Ingestion 设计。

---

# 0. 本版冻结决策

## 0.1 三个主实验

本阶段只跑三条主实验：

```text
Exp A
Fixed Hybrid
S + T + B
→ RRF
→ Top-5
```

```text
Exp B
Fixed Hybrid
S + T + B
→ RRF Top-30
→ qwen3.7-text-rerank
→ Top-5
```

```text
Exp C
Agent Planner
→ 动态选择 S / T / B 的非空子集
→ single route 或 RRF
→ Top-30
→ 同一个 qwen3.7-text-rerank
→ Top-5
```

其中：

```text
S = summary_dense
T = text_dense
B = text_sparse / Milvus BM25
```

实验归因：

```text
Exp B - Exp A
→ Reranker Gain

Exp C - Exp B
→ Planner Gain / Route Cost Saving
```

不增加独立 Stage 3，不复制一套 Planner Retriever / Eval Runner。

---

## 0.2 Fixed Hybrid 与 Agent Planner 共用同一执行链

统一：

```text
RetrievalPolicy
      ↓
RetrievalPlan / normalized SearchRequest
      ↓
Unified Retrieval Executor
      ↓
Candidate Pool
      ↓
Same Reranker
      ↓
Same Evaluator
```

只允许 Policy 不同。

禁止：

```text
FixedHybridRetriever
和
AgentRetriever
```

分别复制一整套：

```text
embedding
Milvus search
RRF
rerank
metrics
```

---

## 0.3 Stage 2 Planner 动作空间只保留“选路”

为保证 `Exp C - Exp B` 可以解释为 **Dynamic Retrieval Policy** 的贡献，本阶段 Planner 的主动作只允许：

```text
选择哪些 Retrieval Channel：
- summary_dense
- text_dense
- text_sparse
```

至少选择一路。

允许结果：

```text
S
T
B
S+T
S+B
T+B
S+T+B
```

但本阶段**不需要把这 7 种 action 逐个离线做完整 benchmark**。

Planner 不自由决定：

```text
collection_name
metric_type
HNSW ef
BM25 low-level params
RRF k
per-route Top-K
reranker Top-K
```

本阶段统一固定：

```text
per_route_k = 50
rrf_k       = 60
candidate_k = 30
final_k     = 5
```

这样：

> Agent 的实验变量主要就是“选择哪些 Channel”，而不是把 Route Choice、Top-K、ANN 参数同时混在一起。

---

## 0.4 Source Filter 不进入 Planner 主动作空间

Stage 1 的：

```text
source_filter_qa
source_filter_literature
```

属于 Smoke / Diagnostic Route，不是新的原子 Retriever。

Stage 2 Planner 主实验不动态选择：

```text
source == qa
source == literature
```

以避免把“选 Retrieval Signal”和“先验知道语料来源”混成一个变量。

现有 filter 能力保留，但不进入 Exp C 主动作空间。

---

# 1. Codex 开始前必须只读检查

## 1.1 我方仓库

必须阅读：

```text
README.md
environment.yml

src/MedicalRag/config/models.py
src/MedicalRag/config/app_config.yaml
src/MedicalRag/config/loader.py

src/MedicalRag/core/KnowledgeBase.py
src/MedicalRag/core/HybridRetriever.py
src/MedicalRag/core/DBFactory.py
src/MedicalRag/core/utils.py

src/MedicalRag/agent/SearchGraph.py
src/MedicalRag/agent/tools/AgentTools.py
src/MedicalRag/prompts/templates.py

src/MedicalRag/ingestion/models.py
src/MedicalRag/ingestion/representations.py

scripts/03_search_data.py

artifacts/stage1/prepare_report.json
artifacts/stage1/ingestion_report.json
artifacts/stage1/smoke_search.json

docs/stage1_ingestion_implementation.md
docs/stage1_ingestion_interview_guide.md
```

重点定位我方已有逻辑：

### Retrieval Schema

```text
src/MedicalRag/config/models.py
```

重点：

```text
SearchRequest
SingleSearchRequest
FusionSpec
AnnsField
```

### Planner / Tool Calling

```text
src/MedicalRag/agent/SearchGraph.py
```

重点：

```text
bind_tools(...)
llm_db_search(...)
call_db prompt 的调用位置
```

```text
src/MedicalRag/agent/tools/AgentTools.py
```

重点：

```text
database_search(search_config: SearchRequest)
```

### Deterministic Retrieval Executor

```text
src/MedicalRag/core/KnowledgeBase.py
```

重点：

```text
_encode_query()
_single_search()
search()
AnnSearchRequest
RRFRanker
WeightedRanker
```

### Planner Prompt

```text
src/MedicalRag/prompts/templates.py
```

重点：

```text
call_db
```

---

## 1.2 参考仓库代码位置

只读参考：

```text
WYW-1127/medicalrag
```

不得覆盖我方架构。

### Reranker Provider

```text
backend/app/core/providers/reranker.py
```

重点参考：

```text
RerankerResult(index, score)
RerankerProvider.rerank()
httpx AsyncClient
timeout
tenacity retry
API error handling
```

注意：

> 参考仓库使用 SiliconFlow 风格 `/rerank` 请求；我方必须改成 `qwen3.7-text-rerank` 的 DashScope / Model Studio 请求结构，不能直接复制 endpoint 或 JSON body。

### Reranker / Retrieval 配置组织

```text
backend/app/core/config.py
```

重点参考：

```text
RerankerSettings
RetrievalSettings

recall_k = 50
rerank_k = 30
rrf_k = 60
```

我方最终：

```text
final_k = 5
```

不照搬参考仓库 `top_k=8`。

### 两阶段 Retriever

```text
backend/app/rag/retriever.py
```

重点参考：

```text
retrieve()
dense/sparse recall
fusion
fused[:rerank_k]
_rerank_chunks()
StageTiming
```

我方差异：

```text
2 routes
→ 改成 3 atomic channels：
summary_dense / text_dense / BM25

Fixed Policy
+
Agent Policy
→ 共用同一 Executor
```

### RRF 实现

```text
backend/app/rag/fusion.py
```

重点参考：

```text
rrf_fuse()
跨 route 按 chunk id 去重
rank-level fusion
fused_score
```

若我方继续使用 Milvus `RRFRanker` 完成线上 Fusion：

> 不必为了代码形式一致改成手写 RRF。

但 Evaluation / Raw Route Offline Analysis 可参考该文件的纯函数实现。

### Retrieval Metrics

```text
backend/app/evaluation/metrics.py
```

重点参考：

```text
recall_at_k()
mrr()
ndcg_at_k()
```

我方需要扩展：

```text
nDCG 支持 rel=2 / 1 / 0 graded relevance
```

### Evaluation Runner

```text
backend/app/evaluation/runner.py
```

重点参考：

```text
run_retrieval_eval()
RETRIEVAL_CONFIGS
统一 runner 执行多配置
按 Query 累积 metrics
```

我方实验配置不是参考仓库的三配置，而是：

```text
Exp A Fixed Hybrid
Exp B Fixed Hybrid + Reranker
Exp C Agent Planner + Same Reranker
```

---

# 2. 执行环境

开始：

```bash
cd <FELIN0366/medical-rag>
conda activate rag
```

报告：

```bash
pwd
git status --short
git rev-parse HEAD
which python
python --version
python -m pip show pymilvus
python -m pip show httpx
docker ps --filter name=milvus-standalone
```

检查：

```bash
python - <<'PY'
import os
print("DASHSCOPE_API_KEY:", bool(os.getenv("DASHSCOPE_API_KEY")))
print("DASHSCOPE_WORKSPACE_ID:", bool(os.getenv("DASHSCOPE_WORKSPACE_ID")))
print("DEEPSEEK_API_KEY:", bool(os.getenv("DEEPSEEK_API_KEY")))
PY
```

不打印 Key 本身。

---

# 3. Stage 1 基线必须先冻结

本阶段不重新：

```text
Parser
Chunk
Corpus
Milvus Schema
Document Embedding
```

确认 Stage 1 当前 Collection：

```text
medical_knowledge
```

已经由：

```text
qwen3.7-text-embedding-flash
```

生成真实 1024D：

```text
summary_dense
text_dense
```

并且：

```text
text_sparse
=
Milvus Built-in BM25
```

如果当前 semantic embedding 重建仍未完成：

```text
BLOCKED
```

等待其完成。

Stage 2 不偷偷使用 local_hash Corpus 继续正式实验。

---

# 4. Reranker 模型与配置——正式冻结

## 4.1 模型

固定：

```text
qwen3.7-text-rerank
```

用途：

```text
Text Search / RAG Candidate Reranking
```

当前官方能力：

```text
最多 500 documents / request
单 query/document 最大 30,000 tokens
推荐单 request 总输入不超过 120,000 tokens
支持 instruct
```

Stage 2：

```text
30 docs / request
```

远低于上限。

---

## 4.2 Provider

使用：

```text
Alibaba Cloud Model Studio / DashScope
```

API Key：

```text
DASHSCOPE_API_KEY
```

Workspace：

```text
DASHSCOPE_WORKSPACE_ID
```

Region：

```text
cn-beijing
```

---

## 4.3 Endpoint

`qwen3.7-text-rerank` 不使用当前 Embedding 的：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

使用：

```text
POST
https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/
api/v1/services/rerank/text-rerank/text-rerank
```

运行时从：

```text
DASHSCOPE_WORKSPACE_ID
```

构造 URL。

不要把真实 Workspace ID 写死进 Git。

---

## 4.4 Request Body

冻结为：

```json
{
  "model": "qwen3.7-text-rerank",
  "input": {
    "query": "<query>",
    "documents": [
      "<candidate 1>",
      "<candidate 2>"
    ]
  },
  "parameters": {
    "top_n": 30,
    "instruct": "Given a medical question, rank passages by whether they contain evidence that directly answers the question. Prefer directly supporting passages over passages that are only topically related."
  }
}
```

不需要：

```text
return_documents
```

本地已经保留 Candidate 文本与 metadata，只需 API 返回：

```text
index
relevance_score
```

再按 index 映射回原 Candidate。

---

## 4.5 Response

`qwen3.7-text-rerank` 读取：

```text
output.results[]
```

每项：

```text
index
relevance_score
```

构造：

```python
class RerankResult(BaseModel):
    index: int
    score: float
```

注意：

> `relevance_score` 只在同一个 request 内用于相对排序，不设计跨 Query 全局 threshold。

---

## 4.6 Reranker Config

建议在：

```text
src/MedicalRag/config/models.py
```

增加：

```python
class RerankerConfig(BaseModel):
    provider: Literal["dashscope"] = "dashscope"
    model: str = "qwen3.7-text-rerank"
    api_key_env: str = "DASHSCOPE_API_KEY"
    workspace_id_env: str = "DASHSCOPE_WORKSPACE_ID"
    region: str = "cn-beijing"
    top_n: int = 30
    timeout: float = 30.0
    max_retries: int = 3
    instruct: str = (
        "Given a medical question, rank passages by whether they contain evidence "
        "that directly answers the question. Prefer directly supporting passages "
        "over passages that are only topically related."
    )
```

在：

```text
src/MedicalRag/config/app_config.yaml
```

增加：

```yaml
reranker:
  provider: dashscope
  model: qwen3.7-text-rerank
  api_key_env: DASHSCOPE_API_KEY
  workspace_id_env: DASHSCOPE_WORKSPACE_ID
  region: cn-beijing
  top_n: 30
  timeout: 30.0
  max_retries: 3
  instruct: >-
    Given a medical question, rank passages by whether they contain evidence
    that directly answers the question. Prefer directly supporting passages
    over passages that are only topically related.
```

不要把：

```text
base_url = embedding compatible-mode/v1
```

复制给 Reranker。

---

# 5. Stage 2 Retrieval 参数——正式冻结

增加统一配置，例如：

```python
class RetrievalStage2Config(BaseModel):
    per_route_k: int = 50
    rrf_k: int = 60
    candidate_k: int = 30
    final_k: int = 5
```

最终：

```text
per_route_k = 50
rrf_k       = 60
candidate_k = 30
final_k     = 5
```

不做 sweep。

---

# 6. Policy Interface

新增或等价实现：

```text
src/MedicalRag/retrieval/policy.py
```

建议：

```python
class RetrievalPolicy(Protocol):
    async def plan(self, query: str) -> SearchRequest:
        ...
```

实现：

```text
FixedHybridPolicy
AgentPlannerPolicy
```

---

# 7. FixedHybridPolicy

固定输出三路：

```text
summary_dense
text_dense
text_sparse
```

每路：

```text
limit = 50
```

Dense：

```text
metric = COSINE
```

Sparse：

```text
metric = BM25
```

Fusion：

```text
RRF(k=60)
```

Executor 输出：

```text
Candidate Top-30
```

---

# 8. AgentPlannerPolicy

复用我方原有 Structured Retrieval Planner。

不得新建一个与现有 Planner 平行的 LLM Agent。

必须复用：

```text
SearchRequest
SingleSearchRequest
FusionSpec
database_search Tool Schema
现有 Planner LLM 配置
call_db Prompt
```

### Planner 输出内容

LLM 只需要决定：

```text
selected anns_field(s)
```

即：

```text
summary_dense
text_dense
text_sparse
```

### Normalization

Planner Tool Call 生成后，统一调用：

```text
normalize_plan()
```

将：

```text
collection_name
metric_type
search_params
limit
fusion k
final candidate limit
```

覆盖为 Stage 2 冻结参数。

例如：

```text
summary_dense
→ COSINE / ef=64 / limit=50

text_dense
→ COSINE / ef=64 / limit=50

text_sparse
→ BM25 / drop_ratio_search=0 / limit=50

multi-route
→ RRF(k=60)

candidate_k
→ 30
```

从而保证：

> Planner 负责选择 Retrieval Signal，不负责调数据库参数。

---

# 9. Planner Prompt

在现有：

```text
src/MedicalRag/prompts/templates.py
```

的：

```text
call_db
```

基础上修改/增加 Stage 2 Planner Prompt。

明确告诉模型：

```text
summary_dense:
适合问题意图、主题、章节语义

text_dense:
适合需要正文细节、语义证据的查询

text_sparse:
适合药名、疾病名、缩写、数值、检查项等 lexical exact matching
```

要求：

```text
选择最少但足够的 Channel
```

不要告诉它 Evaluation Gold 或 raw route result。

---

# 10. Unified Retrieval Executor

建议：

```text
src/MedicalRag/retrieval/executor.py
```

但若当前：

```text
KnowledgeBase.search()
```

已经足以作为统一 Executor：

> 优先扩展现有逻辑，不为了目录而重写。

统一负责：

```text
SearchRequest
→ query encoding
→ selected route search
→ optional RRF
→ Candidate Top-30
```

Fixed / Agent 都走这里。

---

# 11. Query Embedding 只计算一次

当前：

```text
summary_dense
text_dense
```

均为：

```text
qwen3.7-text-embedding-flash
1024D
```

Query 输入相同。

因此 Stage 2：

```text
q = embed(query)
```

一次。

同一个：

```text
q
```

用于：

```text
summary_dense search
text_dense search
```

不要请求两次 DashScope。

---

# 12. Eval Query Embedding Batch

构建所有 Eval Query 后：

```text
batch size = 20
```

批量 Embedding。

缓存：

```text
artifacts/stage2/query_embeddings.*
```

Fixed 与 Agent 使用同一份。

如果保存二进制更合理：

```text
query_embeddings.npz
+
query_embeddings_meta.json
```

即可。

---

# 13. Reranker Client

建议：

```text
src/MedicalRag/retrieval/reranker.py
```

参考：

```text
WYW-1127/medicalrag/backend/app/core/providers/reranker.py
```

保留其：

```text
Async httpx
timeout
transport retry
RerankResult(index, score)
```

但请求必须按本 Spec 第 4 节实现。

API：

```python
async def rerank(
    query: str,
    documents: list[str],
    top_n: int = 30,
) -> list[RerankResult]:
    ...
```

一个 Query：

```text
最多 30 documents
→ 1 HTTP request
```

禁止：

```text
1 document
→ 1 request
```

---

# 14. Reranker 输入

使用：

```text
Document.page_content
```

即 Stage 1 的：

```text
Chunk.text
```

QA：

```text
问题 + 答案
```

Literature：

```text
标题 + section_path + body
```

不另外构造一套 Reranker 文本。

---

# 15. Evaluation Set A：QA Internal

数量：

```text
30
```

从 Stage 1 已入库 200 QA 中固定 seed 抽样。

不直接使用原 Question。

使用：

```text
deepseek-flash
```

生成 user-style paraphrase。

建议：

```text
10 questions / request
```

要求：

```text
保持原医学意图
不增加新医学事实
不复制原句
输出严格 JSON
```

每条：

```json
{
  "query_id": "...",
  "original_question": "...",
  "query": "...",
  "gold_pk": "..."
}
```

---

# 16. Evaluation Set B：Literature

数量：

```text
20
```

从 Stage 1 Literature Chunk 构造。

尽量覆盖：

```text
semantic
lexical
numeric_or_table
topic_or_section
```

每条：

```json
{
  "query_id": "...",
  "query": "...",
  "query_type": "semantic",
  "relevant_pks": ["..."],
  "partial_relevant_pks": ["..."],
  "hard_negative_pks": ["..."],
  "gold_source_name": "...",
  "gold_section_path": "...",
  "gold_evidence_text": "..."
}
```

优先 Gold 质量，不为了 20 条强行使用含糊 Evidence。

---

# 17. Evaluation Set C：Cross-corpus Diagnostic

从当前已发现的：

```text
QA → Literature semantic high-similarity
```

候选中人工核验：

```text
5~10
```

只有：

```text
Literature Evidence 能直接回答 QA Query
```

才保留。

此 Set：

```text
不进入主平均指标
```

用于分析：

```text
direct evidence
vs
topically related hard negative
```

以及：

```text
Planner 选路
Reranker 排序
```

---

# 18. Relevance Label

冻结：

```text
2 = direct evidence
1 = partial relevant
0 = irrelevant / hard negative
```

Recall / MRR：

```text
rel=2
```

视作 Gold。

nDCG：

```text
2 / 1 / 0
```

做 graded relevance。

---

# 19. Evaluation Mode Raw Routes

为了失败归因，Evaluation 模式预先缓存：

```text
S Top-50
T Top-50
B Top-50
```

到：

```text
artifacts/stage2/raw_routes.jsonl
```

用途：

```text
Fixed Hybrid
Atomic route diagnostics
Planner loss attribution
```

这些结果：

> 不能放入 Planner Prompt，不能影响 Agent 决策。

---

# 20. Online Planner Mode

真实 Agent Policy 执行：

```text
Planner 选择 S/B
→ 只查询 S/B
```

不能：

```text
后台实际三路全查
但只假装 Planner 选两路
```

否则：

```text
avg_routes/query
retrieval cost
```

失去意义。

---

# 21. Exp A：Fixed Hybrid

```text
S Top-50
T Top-50
B Top-50
↓
RRF(k=60)
```

保存：

```text
Top-30
```

Final：

```text
RRF Top-5
```

指标：

```text
Candidate Recall@30
Recall@5
MRR
nDCG@5
```

---

# 22. Exp B：Fixed Hybrid + Reranker

复用 Exp A 的：

```text
RRF Top-30
```

调用：

```text
qwen3.7-text-rerank
```

返回完整 Top-30 排序。

最终：

```text
Top-5
```

指标：

```text
Recall@5
MRR
nDCG@5
Rescue@5
Damage@5
Mean Best-Gold Rank Gain
rerank latency p50/p95
```

---

# 23. Exp C：Agent Planner + Same Reranker

```text
Query
↓
AgentPlannerPolicy
↓
selected channel subset
↓
Unified Executor
↓
single route / RRF
↓
Top-30
↓
same qwen3.7-text-rerank
↓
Top-5
```

指标：

```text
Candidate Recall@30
Recall@5
MRR
nDCG@5

avg_routes_per_query
1-route %
2-route %
3-route %

planner p50/p95
retrieval p50/p95
```

---

# 24. Reranker 指标定义

## Rescue@5

```text
Exp A Top-5 没有 direct Gold
Exp B Top-5 有 direct Gold
```

比例。

## Damage@5

```text
Exp A Top-5 有 direct Gold
Exp B Top-5 没有 direct Gold
```

比例。

## Mean Best-Gold Rank Gain

对 Candidate Top-30 内存在 direct Gold 的 Query：

```text
best_gold_rank_before
-
best_gold_rank_after
```

求平均。

---

# 25. Planner 指标解释

Fixed：

```text
routes/query = 3
```

Agent：

```text
routes/query = 实际选中的 channel 数
```

如果：

```text
Agent Quality ≈ Fixed + Reranker
但 avg routes/query < 3
```

则 Planner 仍然体现：

```text
更少 Retrieval Work
```

如果：

```text
Agent Quality > Fixed + Reranker
```

则说明：

```text
动态选路还减少了无关 Candidate 干扰
```

---

# 26. Failure Analysis

必须自动保存：

```text
Top 5 Reranker Rescue
Top 5 Reranker Damage
Top 5 Planner Win
Top 5 Planner Loss
Top 5 BM25-only strong case
Top 5 Dense strong case
```

Planner Loss 必须包含：

```text
selected_channels
gold_rank_summary
gold_rank_text
gold_rank_bm25
```

以回答：

```text
Gold 本来在哪一路？
Planner 为什么没拿到？
```

---

# 27. 主实验表

QA / Literature 分开报告。

## QA

| Config | Candidate Recall@30 | Recall@5 | MRR | nDCG@5 | Avg Routes |
|---|---:|---:|---:|---:|---:|
| Fixed Hybrid | | | | | 3.0 |
| Fixed Hybrid + Reranker | same candidate | | | | 3.0 |
| Agent Planner + Reranker | | | | | |

## Literature

同表。

---

# 28. 不做的评测

本阶段不做：

```text
7 个 Channel subset 全排列 benchmark
Reranker model zoo
candidate_k sweep
HNSW 参数 sweep
Chunk 参数 sweep
Answer Generation Accuracy
RAGAS Full Agent Eval
OOD Refusal
Citation
Safety
```

只做能支持：

```text
Reranker Gain
Planner Gain
Route Cost
```

的关键指标。

---

# 29. 建议代码结构

若不与当前仓库冲突，新增：

```text
src/MedicalRag/retrieval/
├── __init__.py
├── models.py
├── policy.py
├── executor.py
└── reranker.py

src/MedicalRag/evaluation/
├── __init__.py
├── retrieval_dataset.py
├── retrieval_metrics.py
└── retrieval_runner.py
```

但原则：

> 优先复用现有 `KnowledgeBase.py / SearchRequest / SearchGraph / AgentTools`，不要为了目录形式复制旧逻辑。

---

# 30. Scripts

建议新增：

```text
scripts/09_prepare_stage2_eval.py
scripts/10_eval_stage2_retrieval.py
```

### `09_prepare_stage2_eval.py`

负责：

```text
QA 30 fixed sample
DeepSeek batch paraphrase
Literature 20 candidate/eval generation
Cross-corpus diagnostic candidate
Gold / relevance 数据文件
```

不要写 Milvus。

### `10_eval_stage2_retrieval.py`

统一执行：

```text
raw route cache
Exp A
Exp B
Exp C
metrics
latency
failure analysis
artifacts
```

不要分别创建：

```text
10_eval_reranker.py
11_eval_planner.py
```

两套 Runner。

---

# 31. Artifacts

统一：

```text
artifacts/stage2/
├── eval_qa.jsonl
├── eval_literature.jsonl
├── eval_cross_corpus.jsonl
├── query_embeddings_meta.json
├── query_embeddings.npz
├── raw_routes.jsonl
├── fixed_hybrid_predictions.jsonl
├── fixed_rerank_predictions.jsonl
├── agent_rerank_predictions.jsonl
├── metrics.json
├── reranker_analysis.json
├── planner_analysis.json
├── failure_cases.jsonl
├── latency.json
└── config_snapshot.yaml
```

---

# 32. 技术文档

完成后生成：

```text
docs/stage2_retrieval_reranker_planner_implementation.md
```

必须采用“先总后分”：

```text
1. Stage 2 完整 Pipeline
2. Reranker 配置与 API
3. Policy / Executor 解耦
4. Fixed Hybrid
5. Agent Planner
6. Reranker
7. Eval Dataset
8. Metrics
9. Exp A/B/C
10. Reranker Gain
11. Planner Gain
12. Route Cost
13. Failure Analysis
14. API batching/cache
15. 实际实验结果
16. Known Limitations
```

只记录真实运行数字。

---

# 33. 面试文档

生成：

```text
docs/stage2_retrieval_reranker_planner_interview.md
```

至少覆盖：

```text
1. Retriever / RRF / Reranker / Planner 分别解决什么问题？
2. 为什么 Fixed Hybrid 是强 Baseline？
3. 为什么固定 Top-50 → Top-30 → Top-5？
4. 为什么 Reranker 救不回 Candidate Pool 外的 Gold？
5. qwen3.7-text-rerank 为什么适合本项目？
6. 为什么使用 instruct？
7. 为什么 relevance_score 不做全局 threshold？
8. 为什么 Exp B-A 可以解释 Reranker Gain？
9. 为什么 Exp C-B 可以解释 Planner Gain？
10. 为什么 Planner 本阶段只选择 Channel？
11. 为什么不让 Planner 改 ef / metric / RRF k？
12. Eval Mode 为什么预跑 S/T/B？
13. Online Mode 为什么只执行 Planner 选择 route？
14. Planner Failure 如何归因？
15. 为什么需要 routes/query？
16. 为什么 QA 和 Literature 分开统计？
17. partial relevant / hard negative 有什么价值？
18. 为什么 Stage 2 不做 7-action 全量实验？
19. 为什么不做 Reranker model zoo？
20. Stage 2 如何为 Evidence Grade / Citation 铺路？
```

每题：

```text
标准回答
继续追问
代码位置
```

---

# 34. Definition of Done

- [ ] Stage 1 Collection 不重建
- [ ] `qwen3.7-text-rerank` 配置完成
- [ ] DashScope rerank endpoint 真实调用成功
- [ ] FixedHybridPolicy
- [ ] AgentPlannerPolicy
- [ ] Unified Executor
- [ ] Query embedding 复用
- [ ] QA Eval 30
- [ ] Literature Eval 20
- [ ] Cross-corpus Diagnostic 5~10
- [ ] Exp A
- [ ] Exp B
- [ ] Exp C
- [ ] Candidate Recall@30
- [ ] Recall@5
- [ ] MRR
- [ ] graded nDCG@5
- [ ] Rescue@5
- [ ] Damage@5
- [ ] Mean Rank Gain
- [ ] routes/query
- [ ] planner/reranker latency
- [ ] Query-level predictions
- [ ] Failure cases
- [ ] Stage 2 artifacts
- [ ] Implementation doc
- [ ] Interview doc
- [ ] 未进入 Evidence Grade / Citation / Memory / Safety

完成后停止。
