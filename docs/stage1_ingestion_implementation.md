# Stage 1 多源入库实现说明

## 实际执行环境与语料

本实现已在 WSL Ubuntu 的 `rag` Conda 环境中执行，Python 为 3.10.21，解释器路径为
`/home/felin/miniconda3/envs/rag/bin/python`。全程仅使用仓库内已提供的 `./data` 数据，未下载数据集。

实际执行的可复现命令如下：

```bash
python scripts/01_prepare_corpus.py --qa-sample-size 200 --seed 42 --strategy structural
python scripts/02_ingest_data.py --qa-sample-size 200 --seed 42 --strategy structural --recreate
python scripts/03_search_data.py --smoke
```

本次扫描得到 HuatuoQA 检索单元 200 条、PDF 3 个、本地 HTML 1 个、Markdown 1 个（`data/README.md`）。结构化解析共得到 5 份文档、152 个文献 Chunk，最终共写入 352 行 Milvus 数据。`data/eval/` 下两个非 HuatuoQA JSONL 文件均在 inventory 中标为 `ignored_unsupported`，没有被解析或入库。本数据集没有解析失败或 warning。prepare 耗时 2.594 秒，成功的 ingest 耗时 5.823 秒。

## 流水线与表示构造

`src/MedicalRag/ingestion/registry.py` 只发现 Stage 1 支持的数据源：指定的 HuatuoQA JSONL、PDF、HTML 与 Markdown。其他文件仅登记为 `ignored_unsupported`；没有 DOCX、通用 JSON、药品 JSON、CSV、ICD 或结构化表格 Parser。

轻量中间表示位于 `ingestion/models.py`，由 `ParsedDocument` 和递归的 `Section` 构成。`tree.py` 负责构建并遍历章节路径。Parser 不做 embedding，也不写入 Milvus：

- `parsers/pdf.py` 使用 PyMuPDF 的 `get_text("dict")`，估计正文基准字号，以正文的 1.15 倍识别标题，采用简单双栏阅读顺序启发式，将 `find_tables()` 结果转为 Markdown 表格 Section，并为扫描件/无文本页记录 warning。本阶段不做 OCR。
- `parsers/html.py` 仅对本地 HTML 使用 trafilatura 提取正文；若提取结果不含标题，则恢复原始 H1–H6 层级，随后复用 Markdown Parser。提取失败会产生 warning，不会静默丢弃文档。
- `parsers/markdown.py` 支持标题、连续管道表格和代码围栏，因此代码块中的 `#` 不会被误判为标题。

`chunking.py` 提供 structural（默认）、fixed 与 recursive 三种策略。structural 仅在同一章节路径内切分，在 `。；！？\n` 处分句，目标长度为 600 字符，使用 80 字符同章节 overlap；长度不足 100 的尾块会在不超过 600 字符时合并，连续长文本会硬切。长度不超过 4000 字符的表格整体保留，超长表按行拆分且重复表头。

HuatuoQA 不做 Chunk：一个采样 QA 对应一个检索单元，`summary=question`、`document=answer`、`text="问题…答案…"`。文献 Chunk 使用确定性的 `summary=title + section_path + lead text`，当前块正文作为 `document`，标题/章节/正文拼成 `text`。不使用 LLM Summary，因此结果可复现且没有入库阶段的模型成本。主键为 `doc_id + chunk_id + text` 的稳定 SHA-256 哈希。

## 存储与检索

`core/KnowledgeBase.py` 创建冻结后的核心字段：`pk`、text/summary/document、来源元数据、文档/章节元数据、1024 维 `summary_dense` 与 `text_dense`、以及 `text_sparse`。写入 VARCHAR 时使用 `ingestion/representations.py` 中的 UTF-8 字节长度保护，避免将字符数误当作 Milvus 的字节数。

稀疏字段仅由 Milvus 的 `FunctionType.BM25` 从 `text` 生成，并配置 `jieba` 分词器。已删除 Vocabulary、vocab 文件、`BM25Vectorizer` 和自定义稀疏向量主路径。KnowledgeBase 的维度检查现在真正比较 summary 与 text 的维度；Milvus 托管 BM25 的查询分支直接传入查询文本，不再存在原先未定义的 `data = data` 逻辑。

Milvus 使用 Docker standalone 服务 `http://localhost:19530`，dense 索引保持 HNSW/COSINE、1024 维、M=32、efConstruction=200。由于 Ollama 服务和远程 embedding 凭据均不可用，dense 向量使用本地确定性的 1024 维哈希 embedding，避免将本地医疗数据发送至未配置的远程端点。它适合离线 Smoke Test，不应替代生产语义模型。

## 已验证结果

所有产物位于 `artifacts/stage1/`。`ingestion_report.json` 证明 collection 存在，且有 352 行、两个 dense 字段均为 1024 维。`smoke_search.json` 记录了真实的 `summary_dense`、`text_dense`、Milvus BM25、hybrid RRF 与 source filter 检索。QA 路由返回 `source=qa`；高血压查询的 BM25 与 hybrid 路由返回 `source=literature`；两个来源过滤路由只返回各自指定的来源。

已知限制包括：复杂 PDF 版式仍为启发式处理、不含 OCR、HTML 正文质量依赖本地标记、离线哈希 embedding 较为简单。Stage 1 不包含 Reranker 或任何 Stage 2 功能。
