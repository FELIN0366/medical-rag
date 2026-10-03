# Stage 1 入库面试指南

每题均给出标准回答、常见追问和代码位置，面向 Agent/RAG 岗位。

1. **原来的 HuatuoQA 怎么存，现在怎么存？** 标准回答：旧版是 QA 专用 LangChain Document 加自管稀疏向量；新版一个 QA 对应一个 `Chunk`，与文献共用核心 Schema。常见追问：为何改？可共用过滤与检索链路。代码：`ingestion/representations.py`、`core/KnowledgeBase.py`。
2. **HuatuoQA 和文献如何共存？** 标准回答：二者写入同一组字段，通过 `source=qa` 或 `source=literature` 区分。常见追问：QA 有伪章节吗？没有，title/path 为空、page 为 0。代码：`models.py`、`representations.py`。
3. **为什么只用一个 Collection？** 标准回答：同一个查询可融合三路召回，并直接按来源过滤，无需应用侧合并结果。常见追问：以后能拆吗？可以按元数据或独立 Collection 扩展。代码：`KnowledgeBase.py`。
4. **为什么保留三路表示？** 标准回答：summary dense 表达问题/章节意图，text dense 表达上下文，BM25 保留术语精确匹配。常见追问：如何融合？Smoke Test 使用 RRF。代码：`scripts/03_search_data.py`。
5. **QA 的 summary 如何产生？** 标准回答：直接使用 question，不经过生成模型。常见追问：原因？它就是最直接的检索意图。代码：`make_qa_chunk`。
6. **文献的 summary 如何产生？** 标准回答：title、section path 与首句或前 150 字组合。常见追问：可复现吗？完全确定性。代码：`make_literature_chunk`、`lead_text`。
7. **为什么不用 LLM Summary？** 标准回答：保证可复现、避免成本和模型版本波动，也让后续消融更干净。常见追问：以后能加吗？可以作为独立版本实验。代码：`representations.py`。
8. **PDF heading 如何识别？** 标准回答：计算正文中位字号，短文本达到正文 1.15 倍以上时映射为 H1/H2/H3。常见追问：准确吗？它是启发式，失败时退化为普通 Section。代码：`parsers/pdf.py`。
9. **双栏 PDF 如何处理？** 标准回答：中位行宽较窄时按左栏再右栏的顺序排序。常见追问：局限？复杂版面仍需要专门布局模型。代码：`parsers/pdf.py`。
10. **PDF 表格如何处理？** 标准回答：`page.find_tables()` 转 Markdown `Section(is_table=True)`，并从普通文本排除表格框内文字。常见追问：为什么？避免重复入库。代码：`pdf.py`。
11. **扫描版 PDF 如何处理？** 标准回答：无文本层时记录 warning，本阶段明确不做 OCR。常见追问：下一步？在 Parser 前接 OCR。代码：`pdf.py`。
12. **HTML 为什么先提取正文？** 标准回答：导航和模板噪声不应成为医疗知识。常见追问：支持抓取 URL 吗？不支持，只读取本地文件。代码：`parsers/html.py`。
13. **Markdown 如何恢复层级？** 标准回答：H1–H6 通过栈构造树；代码围栏避免误判标题；管道表构成表格 Section。常见追问：树的价值？为 Chunk 提供准确章节路径。代码：`markdown.py`、`tree.py`。
14. **为什么 Parser 与 Chunker 解耦？** 标准回答：Parser 负责语义结构，Chunk 策略可独立替换。常见追问：收益？便于格式复用和消融。代码：`pipeline.py`、`chunking.py`。
15. **Structural Chunk 如何工作？** 标准回答：逐 Section 独立分句并累积到约 600 字符。常见追问：保留什么元数据？每块保留同一 section path 与 page。代码：`chunk_document`。
16. **Fixed、Recursive、Structural 的差异？** 标准回答：fixed 在拍平全文上滑窗；recursive 依次寻找自然分隔符；structural 尊重章节结构。常见追问：默认哪个？structural。代码：`chunking.py`。
17. **overlap 为什么存在？** 标准回答：保留边界处的上下文，使边界信息仍可被检索。常见追问：为什么是 80？这是显式且可比较的基线参数。代码：`_sentence_chunks`。
18. **为什么 overlap 不能跨 Section？** 标准回答：跨章节会让 Chunk 元数据无法如实描述借入文本。常见追问：影响？引用和来源更干净。代码：`chunk_document` 按 Section 独立处理。
19. **short tail 和 hard split 是什么？** 标准回答：短尾块仅在同 Section 且合并后不超过 600 时向前合并；无断句长文本按长度硬切并保留 overlap。常见追问：表格呢？使用专门的按行策略。代码：`chunking.py`。
20. **为什么取消 Custom Vocabulary？** 标准回答：Milvus BM25 托管分词、索引和查询编码，避免维护脆弱的独立语料状态。常见追问：中文分词？Schema 请求 Jieba 分词。代码：`KnowledgeBase.py`。
21. **Milvus BM25 如何工作？** 标准回答：`text` 经 analyzer 后由 BM25 Function 写入 `text_sparse`，稀疏查询直接提交文本。常见追问：如何证明？Smoke Test 的 BM25 路由返回高血压文献。代码：`_encode_query`。
22. **HNSW 与 BM25 各做什么？** 标准回答：HNSW 服务 dense 语义向量；稀疏倒排 BM25 服务词法精确匹配。常见追问：怎么融合？RRF。代码：`build_index`、Smoke 脚本。
23. **为什么 Stage 1 不调 HNSW？** 标准回答：保持 M=32、efConstruction=200，将入库正确性与 ANN 调参隔离。常见追问：何时调优？应在固定语料和评测集上作为独立实验进行。代码：`KnowledgeBase.build_index`。
24. **UTF-8 byte limit 为什么是坑？** 标准回答：Milvus VARCHAR 限制的是字节，中文字符可能超过同样的字符数切片。常见追问：怎么修？编码后按字节截取再忽略非法尾字节解码。代码：`fit_varchar`。
25. **当前实现有哪些限制？** 标准回答：PDF 版式为启发式、没有 OCR、HTML 质量依赖正文提取、Lite 缺少 HNSW、离线哈希 embedding 仅用于 Smoke。常见追问：在哪里说明？实现文档。代码：`docs/stage1_ingestion_implementation.md`。
26. **下一阶段 Reranker 如何接？** 标准回答：从任意 dense/BM25/hybrid 路由取 top-k `Document`，用 query 和 `text` 打分后重排，再交给生成模型。常见追问：需要改入库吗？不需要，元数据和三路表示已保留。代码：`KnowledgeBase.search`。
