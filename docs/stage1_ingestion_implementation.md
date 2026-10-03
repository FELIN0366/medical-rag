# Stage 1 多源 Ingestion 实现说明

## 1. Stage 1 的算法边界

Stage 1 的目标是构建干净、可复现的多源检索基线：HuatuoQA 固定采样、PDF/HTML/Markdown 解析、章节树、三种 Chunking、三路表示、Milvus 单 Collection 与真实 Smoke Test。它不实现 DOCX、药品或通用 JSON、CSV/ICD、文档版本注册、Reranker、Planner、Memory、Citation、安全评估或任何 Stage 2 功能。

HuatuoQA 默认采样 200 条，允许范围为 100～300，使用 `seed=42`。采样由本地 JSONL 完成，既不下载数据，也不默认扫描全部 5 万条进行入库。

## 2. 完整 Data Flow

```text
./data
  ├─ qa_50000.jsonl ─→ 固定采样 ─→ QA Representation
  └─ PDF / HTML / Markdown ─→ Parser ─→ ParsedDocument / Section Tree
                         ─→ Chunker ─→ Literature Representation
                                              ↓
                         summary / document / text + metadata
                                              ↓
       summary_dense / text_dense + Milvus FunctionType.BM25(text)
                                              ↓
                 Docker Milvus Standalone 单 Collection
                                              ↓
          dense / BM25 / RRF hybrid / source filter Smoke Test
```

`data/README.md` 和 `data/eval/**` 是说明或评测数据，不属于医学语料；它们写入 inventory，状态为 `ignored_non_corpus`，不会形成 literature Chunk。其他不支持的文件记录为 `ignored_unsupported`。

## 3. Parser、Section Tree 与 Chunk 为什么解耦

Parser 的职责是将文件转换为轻量 `ParsedDocument`，保留 title、source、warnings 和 `Section` 列表；它不做 embedding、不写数据库，也不决定检索策略。`tree.py` 只维护标题层级与 `section_path`。Chunker 从章节树取得正文、页码和路径，Representation 层再把 Chunk 转成检索字段。这样的四层边界让相同 Chunker 可以复用到 PDF、HTML、Markdown，也使 fixed/recursive/structural 可独立消融。

## 4. PDF Parser 的具体 heuristic

`parsers/pdf.py` 使用 PyMuPDF 的 `page.get_text("dict")`，从 block/line/span 提取文本、字号和 bbox。全篇非表格文本字号的中位数作为正文基准；短文本达到正文的 1.15 倍时为标题候选，并按 1.45、1.28 和 1.15 倍映射至 H1/H2/H3。启发式失败时仍将文本保留为普通 Section，不会使整份 PDF 失败。

对于疑似双栏页面，若中位行宽小于页面宽度的 0.62，则按左栏、右栏以及纵向位置排序。该规则只服务简单双栏，复杂表格、浮动图文和混排版式不保证正确。`page.find_tables()` 的表格转换为 Markdown `Section(is_table=True)`，表格 bbox 与普通正文去重。无文本层或空页记录扫描件 warning；Stage 1 不做 OCR。

## 5. HTML 与 Markdown Parser

HTML Parser 只读取本地文件，先用 trafilatura 提取正文；提取为空时产生 warning 并使用本地文本回退。它收集原始 H1～H6；若 trafilatura 输出未保留标题，再补回标题层级并复用 Markdown Parser。

Markdown Parser 支持 H1～H6、连续管道表格和代码围栏。代码围栏内的 `#` 不会被当成标题；表格被单独标为 `is_table=True`，使后续表格 Chunk 规则可见。

## 6. Structural、Fixed、Recursive 的真实实现

默认 structural 策略逐个 Section 处理，以 `。；！？\n` 分句并累积到约 600 字符；overlap 为 80 字符且只在同一 Section 内产生。尾块小于 100 字符时，仅在同 Section 且合并后不超过 600 字符才向前合并。没有可用句界的长文本执行 hard split。

fixed 策略对拍平后的正文以 600/80 滑窗，不使用章节信息。recursive 策略按 `\n\n → \n → 。 → ； → ，` 优先级切分；某层分隔后的 piece 仍超过 600 字符时，继续使用下一层分隔符；最后无分隔符时 hard split。非表格 recursive Chunk 满足 `len(document) <= max_chars`，由 `tests/test_stage1_recursive_chunking.py` 验证。

表格不套用普通 600 字符规则：不超过 4000 字符时整体保留；超长时按行分片并在后续片段重复表头。

## 7. QA 与 Literature 两种 Representation

HuatuoQA 不 Chunk：一问一答就是一个检索单元。`summary=question`，`document=answer`，`text="问题: question\n\n答案: answer"`，并使用 `source=qa`、`source_name=huatuo_qa`、`chunk_id=0`。

文献 Chunk 的 `summary` 为 `title + section_path + lead text`，其中 lead text 是首句或最多 150 字符；`document` 是当前 Chunk 正文；`text` 为标题、章节路径和正文的确定性拼接。主键是 `stable_hash(doc_id + chunk_id + text)`。Stage 1 不用 LLM 生成摘要，从而避免入库成本和模型版本差异。

## 8. 三路表示的语义

`summary_dense` 嵌入 QA 问题或文献标题/章节/导语，适合较短的意图表示；`text_dense` 嵌入完整 `text`，包含正文上下文；`text_sparse` 不由 Python 计算，而由 Milvus 对 `text` 生成 BM25 稀疏向量。三路字段共用一份 metadata，因此结果可按来源、文档、章节和页码定位。

## 9. Milvus Schema、HNSW 与 BM25

基础设施是 Docker `milvusdb/milvus:v2.6.0` standalone，端点为 `http://localhost:19530`。核心字段为 `pk`、`text`、`summary`、`document`、`source`、`source_name`、`doc_id`、`chunk_id`、`department`、`title`、`section_path`、`page`、`summary_dense`、`text_dense`、`text_sparse`。

两个 dense 字段均为 1024 维 `FLOAT_VECTOR`，索引为 HNSW/COSINE，参数为 `M=32`、`efConstruction=200`。`text` 开启 analyzer，使用 Jieba tokenizer；`FunctionType.BM25` 将其写入 `text_sparse`；稀疏索引为 `SPARSE_INVERTED_INDEX`，度量为 BM25。VARCHAR 写入前通过 UTF-8 字节保护截断，避免中文多字节越界。

## 10. RRF 的位置

单路请求可分别查询 summary dense、text dense 或 BM25。多个 `AnnSearchRequest` 使用 Milvus hybrid search，通过 `RRFRanker(k=60)` 融合排序。RRF 是排名融合方法，不是 Recall、Precision 或语义质量实验；它只验证多个可用召回列表可在同一 Collection 中合并。

## 11. 真实 Dense Embedding 与离线回退

默认 `summary_dense` 与 `text_dense` 使用 DashScope OpenAI-compatible 服务的 `qwen3.7-text-embedding-flash`，配置为 1024 维，端点为 `https://dashscope.aliyuncs.com/compatible-mode/v1`，密钥从 `DASHSCOPE_API_KEY` 读取。前者嵌入 QA 问题或文献摘要，后者嵌入完整检索文本；两者均写入 Docker Milvus 的 HNSW/COSINE 索引。`text_sparse` 仍由 Milvus Jieba Analyzer 与 `FunctionType.BM25` 生成，绝不回退为自管词表或 Python BM25。

`LocalHashEmbeddings` 保留为离线 CI、调试和基础设施 Smoke 回退：它只验证 `text → 1024D vector → HNSW → COSINE search`，不作为默认配置，也不参与正式检索实验。真实 embedding 的 Smoke 可以证明从语义 embedding 到 Milvus dense、BM25、RRF 的完整检索链可运行；但没有 Gold Query/Evidence 标注时，仍不能得出 Dense Recall、字段相对收益或 hybrid 质量提升的结论。

## 12. 当前真实实验结果

清理 `data/README.md` 与 `data/eval/**` 后，使用 `--qa-sample-size 200 --seed 42 --strategy structural` 重新生成 artifacts。准备阶段发现 3 个 PDF、1 个 HTML、0 个可入库 Markdown，另有 3 个 `ignored_non_corpus` 文件；4 份文献解析成功、无解析失败，形成 150 个文献 Chunk 和 200 条 QA，共 350 条检索单元。文献 Chunk 的平均长度为 526.33，P50 为 559.5，P95 为 599，表格 Chunk 为 16 个。

切换默认稠密模型后，既有 `summary_dense` 和 `text_dense` 不会自动更新，因此本次以 `02_ingest_data.py --recreate` 重建了 `medical_knowledge`。DashScope `qwen3.7-text-embedding-flash` 已实际生成两路 1024 维向量；入库报告显示 350 条写入记录、350 条 collection 可见记录，完成耗时 61.021 秒。随后 `03_search_data.py --smoke` 覆盖并通过 summary dense、text dense、text sparse BM25、RRF hybrid、QA 来源过滤、文献来源过滤六条路由；两个 dense 路由返回 QA，BM25、hybrid 与文献过滤返回 literature，来源过滤结果只包含目标 source。`artifacts/stage1/ingestion_report.json` 与 `smoke_search.json` 已是本次真实语义 embedding 的运行证据；准备阶段产物不含 embedding 副作用，无需因模型切换而重跑。

## 13. 当前已知限制

PDF 标题与双栏规则均为启发式，扫描 PDF 不含 OCR；HTML 正文质量依赖 trafilatura；尚未做 Recall、Latency、吞吐量、百万规模或医学问答准确率评测。上述限制是 Stage 1 的明确边界，不应通过引入不在范围内的组件掩盖。

## 14. 与 Stage 2 Reranker 的接口

Stage 2 应以当前 `KnowledgeBase.search()` 返回的 top-k `Document` 作为 Candidate Recall 输入，使用 query 与 `text` 重排，并保留 `pk`、source、doc_id、section_path、page 等元数据。Reranker 只位于召回与生成之间，不改变 Stage 1 Parser、Chunk、Representation 或 Milvus Schema；其效果需要独立评测，而不能从 Smoke Test 推断。
