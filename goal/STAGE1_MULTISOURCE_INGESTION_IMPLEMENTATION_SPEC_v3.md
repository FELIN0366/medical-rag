# Stage 1 Implementation Spec v3
# HuatuoQA + PDF / HTML / Markdown 多源 Ingestion + Chunking + Core Milvus Schema

> **目标仓库（唯一允许修改）**：`FELIN0366/medical-rag`（有短横线）  
> **参考仓库（只读）**：`WYW-1127/medicalrag`  
> **执行环境**：WSL Ubuntu  
> **Conda 环境**：`rag`  
> **数据目录**：仓库根目录 `./data/`，本阶段所需数据均已下载完成，禁止再次联网下载  
> **核心原则**：只保留最新版实现，不做旧 Ingestion / Vocabulary 架构兼容  
> **本阶段支持的数据源**：  
> 1. HuatuoQA（仅固定采样 100～300 条，默认 200）  
> 2. PDF 医学文档  
> 3. HTML 医学文档  
> 4. Markdown 医学文档  
>
> **明确删除 / 不支持**：药品说明书 JSON、CSV / ICD 等结构化表格、DOCX。  
> **阶段边界**：只完成 Parser、Section Tree、Chunking、三路表示构造、Milvus 建库/入库和检索 Smoke Test；不做 Reranker、SearchGraph V2、Memory、安全评测、多模态训练。

---

# 0. 本版冻结决策

## 0.1 不做 Legacy Compatibility

旧版：

```text
scripts/01_build_vocab.py
scripts/02_ingest_data.py
src/MedicalRag/core/IngestionPipeline.py
src/MedicalRag/embed/sparse.py
src/MedicalRag/embed/bm25.py
```

围绕：

```text
HuatuoQA
→ 自建 Vocabulary / BM25
→ QA-specific Ingestion
```

本阶段直接替换为：

```text
scripts/01_prepare_corpus.py
scripts/02_ingest_data.py
scripts/03_search_data.py

src/MedicalRag/ingestion/
    models.py
    tree.py
    registry.py
    chunking.py
    representations.py
    pipeline.py
    parsers/
        pdf.py
        html.py
        markdown.py

src/MedicalRag/core/KnowledgeBase.py
```

当：

```text
prepare → ingest → search smoke
```

全部通过后：

- 删除旧 `scripts/01_build_vocab.py`
- 用新版直接覆盖 `scripts/02_ingest_data.py`
- 更新 `scripts/03_search_data.py`
- 删除旧 `src/MedicalRag/core/IngestionPipeline.py`
- 删除仅服务于旧自管理 Vocabulary/BM25 的代码，如 `embed/sparse.py`、`embed/bm25.py`，但删除前必须 code search 确认新版无引用
- 删除不再使用的旧 Vocabulary/BM25 配置

不保留 wrapper，不保留兼容入口。

---

## 0.2 HuatuoQA 只采样 100～300 条

默认：

```text
qa_sample_size = 200
seed = 42
```

允许范围：

```text
100 <= qa_sample_size <= 300
```

要求：

- 固定 seed；
- 采样可复现；
- 不默认处理 5 万条；
- 技术文档记录实际入库 QA 数量。

---

## 0.3 文档 Parser 只支持 PDF / HTML / Markdown

本阶段只实现：

```text
PDF
HTML
Markdown
```

不实现：

```text
DOCX
Drug JSON
Generic JSON
CSV
ICD
Structured Table
```

HuatuoQA JSON/JSONL 是单独的 QA 数据入口，不经过文档 Parser。

---

## 0.4 Milvus Schema 保持核心化

不做复杂版本治理，不增加：

```text
doc_hash
chunk_hash
chunking_version
source_tier
corpus_scope
published_year
version registry
MySQL document registry
```

本阶段允许：

```text
drop collection
→ create latest schema
→ re-ingest selected corpus
```

---

# 1. Codex 开始前必须只读检查

## 1.1 我方仓库

必须阅读：

```text
README.md
environment.yml
setup.py

scripts/01_build_vocab.py
scripts/02_ingest_data.py
scripts/03_search_data.py

src/MedicalRag/config/models.py
src/MedicalRag/config/app_config.yaml

src/MedicalRag/core/IngestionPipeline.py
src/MedicalRag/core/KnowledgeBase.py
src/MedicalRag/core/insert.py

src/MedicalRag/embed/sparse.py
src/MedicalRag/embed/bm25.py

src/MedicalRag/agent/SearchGraph.py
```

实际扫描：

```bash
find ./data -maxdepth 5 -type f | sort
```

以 `./data` 真实内容为唯一事实。

---

## 1.2 参考仓库代码位置

### 统一 IR / Section Tree

```text
WYW-1127/medicalrag/backend/app/ingestion/models.py
WYW-1127/medicalrag/backend/app/ingestion/tree.py
WYW-1127/medicalrag/backend/app/ingestion/registry.py
```

### PDF

```text
WYW-1127/medicalrag/backend/app/ingestion/parsers/pdf.py
```

### HTML

```text
WYW-1127/medicalrag/backend/app/ingestion/parsers/html.py
```

### Markdown

```text
WYW-1127/medicalrag/backend/app/ingestion/parsers/markdown.py
```

### Chunking

```text
WYW-1127/medicalrag/backend/app/ingestion/chunking.py
```

### Milvus Store / Pipeline

```text
WYW-1127/medicalrag/backend/app/ingestion/store.py
WYW-1127/medicalrag/backend/app/ingestion/pipeline.py
```

### 行为测试参考

```text
WYW-1127/medicalrag/backend/tests/test_chunking.py
WYW-1127/medicalrag/backend/tests/test_parser_pdf.py
WYW-1127/medicalrag/backend/tests/test_store.py
```

仅借鉴设计思想和行为，不直接覆盖目标仓库。

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
which python
python --version
conda env list
find ./data -maxdepth 5 -type f | sort
```

约束：

- 不创建新 Conda 环境；
- 不使用系统 Python；
- 不使用 `sudo pip`；
- 不升级 torch / CUDA / transformers 等无关核心依赖；
- 缺少 `pymupdf`、`trafilatura` 等仅安装到 `rag` 环境并同步依赖声明；
- 不联网下载数据。

---

# 3. Stage 1 最终脚本链

最终只保留：

```text
scripts/
├── 01_prepare_corpus.py
├── 02_ingest_data.py
└── 03_search_data.py
```

---

## 3.1 `01_prepare_corpus.py`

替代：

```text
01_build_vocab.py
```

职责：

```text
scan ./data
→ 固定采样 HuatuoQA
→ 发现 PDF/HTML/MD
→ Parser
→ ParsedDocument
→ Section Tree
→ Chunk
→ Representation Builder
→ 输出统计与样本
```

不：

- 调 Embedding；
- 写 Milvus；
- 构建 vocab。

命令：

```bash
python scripts/01_prepare_corpus.py \
  --qa-sample-size 200 \
  --seed 42 \
  --strategy structural
```

输出：

```text
artifacts/stage1/corpus_inventory.json
artifacts/stage1/prepare_report.json
artifacts/stage1/chunk_samples.jsonl
```

---

## 3.2 `02_ingest_data.py`

完全重写旧脚本。

职责：

```text
读取 HuatuoQA Sample + PDF/HTML/MD
→ Parse
→ Chunk
→ 构造 summary/document/text
→ summary embedding
→ text embedding
→ recreate Milvus
→ insert
→ build/load index
→ report
```

命令：

```bash
python scripts/02_ingest_data.py \
  --qa-sample-size 200 \
  --seed 42 \
  --strategy structural \
  --recreate
```

---

## 3.3 `03_search_data.py`

更新为新版 Schema Smoke Test。

必须测试：

```text
summary_dense Search
text_dense Search
text_sparse / Milvus BM25 Search
hybrid Search
source metadata filter
```

查询至少覆盖：

```text
1 个 HuatuoQA 类型问题
1 个 PDF/HTML/MD 文档类型问题
```

若某种文档格式在 `./data` 实际不存在，不要捏造测试数据源，报告缺失即可。

---

# 4. 数据源分类

最终只存在两大知识来源：

```text
A. qa
   └── HuatuoQA

B. literature
   ├── PDF
   ├── HTML
   └── Markdown
```

其中：

```text
source = "qa"
```

或：

```text
source = "literature"
```

`source_name` 用于标识：

```text
huatuo_qa
某指南名称
某 HTML/Markdown 文档名称
```

---

# 5. 统一 ParsedDocument

新增：

```text
src/MedicalRag/ingestion/models.py
```

保持轻量：

```python
class Section(BaseModel):
    level: int
    title: str
    text: str = ""
    page: int | None = None
    is_table: bool = False
    children: list["Section"] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    doc_id: str
    title: str

    source: str
    source_name: str
    department: str

    sections: list[Section] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

说明：

- PDF/HTML/MD → ParsedDocument；
- HuatuoQA 不强制走 ParsedDocument，可直接进入 QA Representation Builder；
- Parser 不写 Milvus；
- Parser 不做 Embedding；
- Parser 不处理 Retrieval 策略。

---

# 6. 统一 Literature Chunk

```python
class Chunk(BaseModel):
    pk: str

    doc_id: str
    chunk_id: int

    source: str
    source_name: str
    department: str

    title: str
    section_path: str
    page: int | None = None

    summary: str
    document: str
    text: str
```

主键：

```text
pk = stable_hash(doc_id + chunk_id + text)
```

只要求：

- 唯一；
- 确定性；
- 重建 Collection 时稳定。

---

# 7. Core Milvus Schema

最终冻结为：

```text
pk              VARCHAR PRIMARY KEY

text            VARCHAR
summary         VARCHAR
document        VARCHAR

source          VARCHAR
source_name     VARCHAR

doc_id          VARCHAR
chunk_id        INT64

department      VARCHAR
title           VARCHAR
section_path    VARCHAR
page            INT64

summary_dense   FLOAT_VECTOR(1024)
text_dense      FLOAT_VECTOR(1024)
text_sparse     SPARSE_FLOAT_VECTOR
```

相比旧 Schema，核心思想仍然是：

```text
文本字段
+
来源/章节 metadata
+
三路 Retrieval 表示
```

不加入复杂版本字段。

---

# 8. HuatuoQA Representation

HuatuoQA 不做 Chunk：

```text
1 QA Pair = 1 Retrieval Unit
```

构造：

```text
summary = question

document = answer

text =
问题: {question}

答案: {answer}
```

向量：

```text
summary_dense ← embed(summary)
text_dense    ← embed(text)
text_sparse   ← Milvus BM25(text)
```

metadata：

```text
source = qa
source_name = huatuo_qa
doc_id = stable id for QA record
chunk_id = 0
department = ""
title = ""
section_path = ""
page = 0
```

---

# 9. Literature Representation

适用于：

```text
PDF
HTML
Markdown
```

每个 Chunk：

```text
summary =
title
+ section_path
+ lead text
```

其中：

```text
lead text = chunk 首句或前 120～160 chars
```

`document`：

```text
当前 Chunk 原始正文
```

`text`：

```text
文档：{title}
章节：{section_path}
正文：
{document}
```

向量：

```text
summary_dense ← embed(summary)
text_dense    ← embed(text)
text_sparse   ← Milvus BM25(text)
```

---

# 10. 为什么 Literature Summary 不使用 LLM

Stage 1 要求 deterministic：

```text
Summary = deterministic function(title, section_path, chunk)
```

原因：

- 可复现；
- 不增加 Ingestion LLM 成本；
- 不受模型版本影响；
- 后续 Reranker / Planner 消融更干净。

Stage 1 不实现 LLM-generated summary。

---

# 11. PDF Parser

参考：

```text
WYW-1127/medicalrag/backend/app/ingestion/parsers/pdf.py
```

流程：

```text
PyMuPDF
→ page.get_text("dict")
→ block
→ line
→ span
```

提取：

```text
text
font size
bbox
page
```

---

## 11.1 Heading

估计 body font size。

标题候选：

```text
font_size >= body_size * 1.15
```

按较大字号映射 H1/H2/H3。

Heading heuristic 失败时退化为普通 Section，不允许整篇失败。

---

## 11.2 Double-column

可参考：

```text
median(line_width) < 0.62 * page_width
```

进行简单双栏 reading order 恢复。

必须在技术文档说明这是 heuristic，不保证复杂版面。

---

## 11.3 Table

使用：

```text
page.find_tables()
```

表格转 Markdown：

```text
Section(is_table=True)
```

表格 bbox 内文字不能再重复进入普通正文。

---

## 11.4 Scanned PDF

无文本层：

```text
warnings += suspected scanned page
```

本阶段：

```text
不做 OCR
```

---

# 12. Markdown Parser

参考：

```text
WYW-1127/medicalrag/backend/app/ingestion/parsers/markdown.py
```

支持：

```text
#      → H1
##     → H2
...
###### → H6
```

连续 Markdown Table：

```text
| ... |
```

转换：

```text
Section(is_table=True)
```

Code Block 内：

```text
# text
```

不能误判成标题。

---

# 13. HTML Parser

参考：

```text
WYW-1127/medicalrag/backend/app/ingestion/parsers/html.py
```

流程：

```text
本地 HTML
→ trafilatura.extract()
→ 主体 Markdown-like text
→ 恢复原始 h1~h6 层级
→ 复用 Markdown Parser
```

本阶段不联网抓取 URL。

正文抽取失败要：

```text
warning
```

而不是静默丢文档。

---

# 14. 本阶段不要实现 DOCX / Structured Parser

删除上一版设计中的：

```text
parsers/docx.py
parsers/structured.py
```

也不需要安装：

```text
python-docx
```

除非仓库其他现有功能本身依赖它；如果仅 Stage 1 使用，则不要新增。

不处理：

```text
drug json
csv
icd
structured table
```

`./data` 中即使存在这些文件，本阶段 inventory 可以记录为：

```text
ignored_unsupported
```

但不要入库。

---

# 15. Chunking

新增：

```text
src/MedicalRag/ingestion/chunking.py
```

参考：

```text
WYW-1127/medicalrag/backend/app/ingestion/chunking.py
```

默认：

```text
strategy = structural
max_chars = 600
min_chars = 100
overlap = 80
table_max_chars = 4000
```

保留：

```text
fixed
recursive
```

作为未来消融 baseline。

---

# 16. Structural Chunking

```text
Section Tree
→ 遍历每个 Section
→ 构造 section_path
→ 当前 Section 内按句子边界切分
→ 聚合至约 max_chars
→ 输出 Chunk
```

句子边界：

```text
。
；
！
？
\n
```

---

# 17. Overlap

默认：

```text
80 chars
```

只允许发生在：

```text
同一个 section_path
```

禁止：

```text
Section A 尾部
→ overlap 到 Section B
```

---

# 18. Short Tail

尾块：

```text
len < min_chars
```

且：

```text
same section
+
merge 后不超 max_chars
```

则向前合并。

---

# 19. Long Unbroken Text

如果单个 segment：

```text
> max_chars
```

且没有合适句子边界：

```text
hard split
```

并保留同 Section overlap。

---

# 20. Table Chunking

普通表：

```text
<= table_max_chars
→ entire table as one chunk
```

超长表：

```text
按行拆分
```

后续每片重复表头。

---

# 21. Fixed Baseline

```text
拍平全文
→ 600 char sliding window
→ 80 char overlap
```

不使用结构信息。

---

# 22. Recursive Baseline

依次：

```text
\n\n
\n
。
；
，
```

寻找自然边界。

---

# 23. Milvus Built-in BM25

新版 Stage 1 只保留：

```text
Milvus built-in BM25
```

链路：

```text
text VARCHAR
    ↓
Milvus analyzer
    ↓
BM25 Function
    ↓
text_sparse
    ↓
SPARSE_INVERTED_INDEX
```

因此删除旧主链：

```text
Vocabulary
vocab.pkl.gz
BM25Vectorizer
BM25SparseEmbedding
scripts/01_build_vocab.py
```

中文 analyzer 使用当前 Milvus 版本验证可用的 tokenizer，优先参考生产仓库 `jieba` 设置。

---

# 24. Dense Index 保持原项目参数

不改：

```text
summary_dense:
  HNSW
  COSINE
  dim = 1024
  M = 32
  efConstruction = 200

text_dense:
  HNSW
  COSINE
  dim = 1024
  M = 32
  efConstruction = 200
```

Stage 1 不做 ANN 参数调优。

---

# 25. 必须修复原 KnowledgeBase 问题

## 25.1 Dimension Assert

修复原：

```python
summary_dense.dimension == summary_dense.dimension
```

为真正检查：

```text
summary_dense.dimension == text_dense.dimension
```

---

## 25.2 Milvus BM25 Query Branch

核对并修复 `_encode_query()` 中 Milvus-managed BM25 分支。

Dense：

```text
embed_query(query)
```

Sparse：

```text
query text
→ Milvus BM25
```

不能出现未定义：

```python
data = data
```

逻辑。

---

# 26. UTF-8 Byte Guard

Milvus VARCHAR `max_length` 按字节理解。

提供：

```python
def fit_varchar(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
```

不允许只按：

```python
text[:N]
```

假定等于 N bytes。

---

# 27. `01_prepare_corpus.py` 验收

必须输出：

```text
HuatuoQA sampled count
PDF count
HTML count
Markdown count
Ignored unsupported file count
Parse failure count
Parsed document count
Chunk count
Mean chunk chars
P50
P95
Table chunk count
Warnings
```

至少打印：

```text
1~3 个 QA Unit
1~3 个 Literature Chunk
```

包含：

```text
source
source_name
title
section_path
summary
document preview
```

---

# 28. `02_ingest_data.py` 验收

完成后打印：

```text
collection name
qa rows
literature chunks
total rows
summary_dense dim
text_dense dim
sparse BM25 status
elapsed time
```

并检查：

```text
collection exists
row count > 0
```

---

# 29. `03_search_data.py --smoke`

至少验证：

```text
summary_dense
text_dense
text_sparse
hybrid
```

结果输出：

```text
query
route
rank
pk
source
source_name
title
section_path
distance
document preview
```

必须真实证明：

```text
HuatuoQA + Literature
```

能够共存在同一 Collection 中并被检索。

---

# 30. Artifacts

生成：

```text
artifacts/stage1/
├── corpus_inventory.json
├── prepare_report.json
├── chunk_samples.jsonl
├── ingestion_report.json
└── smoke_search.json
```

---

# 31. 技术文档

完成后强制生成：

```text
docs/stage1_ingestion_implementation.md
docs/stage1_ingestion_interview_guide.md
```

---

## 31.1 Implementation 文档

必须写真实结果：

1. WSL / `rag` 环境；
2. `./data` 实际扫描结果；
3. 哪些文件被支持：
   - HuatuoQA
   - PDF
   - HTML
   - Markdown
4. 哪些文件被明确忽略；
5. HuatuoQA 实际 sample size；
6. PDF Parser；
7. HTML Parser；
8. Markdown Parser；
9. Section Tree；
10. Structural / Fixed / Recursive；
11. Chunk 参数；
12. QA summary/document/text；
13. Literature summary/document/text；
14. Milvus Schema；
15. summary_dense / text_dense / BM25；
16. 为什么删除 `01_build_vocab.py`；
17. 修复的 KnowledgeBase bug；
18. 实际 row/chunk 数；
19. Smoke Search 结果；
20. 已知限制。

---

## 31.2 面试文档

至少回答：

1. 原来的 HuatuoQA 如何存 Milvus？
2. 现在 HuatuoQA 和文献如何共存？
3. 为什么一个 Collection？
4. 为什么三路 Representation 不变？
5. QA 的 summary 怎么产生？
6. Literature 的 summary 怎么产生？
7. 为什么不用 LLM Summary？
8. PDF Heading 如何检测？
9. 双栏如何处理？
10. PDF 表格如何处理？
11. 扫描 PDF 如何处理？
12. HTML 为什么先做正文抽取？
13. Markdown 如何恢复层级？
14. 为什么 Parser 与 Chunker 解耦？
15. Structural Chunk 怎么做？
16. Fixed / Recursive / Structural 有什么差异？
17. overlap 为什么存在？
18. 为什么不能跨 Section overlap？
19. short tail 和 hard split 是什么？
20. 为什么取消 Custom Vocabulary？
21. Milvus BM25 如何工作？
22. HNSW 与 BM25 各自负责什么？
23. 为什么 Stage 1 不动 HNSW 参数？
24. UTF-8 byte limit 为什么是坑？
25. 当前实现还有哪些已知限制？
26. 下一阶段 Reranker 如何接在当前 Pipeline 后？

每项包含：

```text
标准回答
常见追问
代码位置
```

---

# 32. 删除清单

新版全部验证成功后检查并删除：

```text
scripts/01_build_vocab.py
src/MedicalRag/core/IngestionPipeline.py
src/MedicalRag/embed/sparse.py
src/MedicalRag/embed/bm25.py
旧 vocab.pkl.gz
旧 Vocabulary/BM25 专属配置
```

删除前：

```text
grep / code search
```

确认没有新版引用。

旧 `02_ingest_data.py`：

```text
直接由新版覆盖
```

旧 `03_search_data.py`：

```text
直接升级为新版 Smoke
```

---

# 33. Definition of Done

- [ ] WSL + `conda activate rag`
- [ ] 正确目标仓库 `FELIN0366/medical-rag`
- [ ] 未联网下载数据
- [ ] HuatuoQA 默认采样 200，seed=42
- [ ] QA sample 可复现
- [ ] PDF Parser 可运行
- [ ] HTML Parser 可运行
- [ ] Markdown Parser 可运行
- [ ] DOCX/Drug JSON/CSV/ICD 不进入新版 Pipeline
- [ ] Section Tree 正常
- [ ] Structural Chunk 正常
- [ ] Fixed / Recursive 可运行
- [ ] 表格保持结构
- [ ] QA Representation 正确
- [ ] Literature Representation 正确
- [ ] Core Milvus Schema 创建成功
- [ ] summary_dense = 1024D
- [ ] text_dense = 1024D
- [ ] text_sparse = Milvus BM25
- [ ] summary_dense Search 成功
- [ ] text_dense Search 成功
- [ ] BM25 Search 成功
- [ ] Hybrid Search 成功
- [ ] dimension assert bug 修复
- [ ] Milvus BM25 query bug 修复
- [ ] 旧 build-vocab / custom sparse 主链删除
- [ ] artifacts/stage1/* 生成
- [ ] 两份技术文档生成
- [ ] 未进入 Stage 2 Reranker

完成后停止。
