# Stage 1 Ingestion 面试指南

本指南只解释 Stage 1 已实现的基线。每题按“标准回答、为什么这样设计、继续追问、陷阱/错误回答、代码位置”组织；不把 Smoke Test 误称为召回率实验。

## 1. QA 和长文档为什么不能完全使用一样的 Ingestion？

- **标准回答：** QA 的自然检索单位是一问一答，HuatuoQA 不再切块；长文档必须先恢复章节和正文边界再切块。
- **为什么这样设计：** QA 再切会拆散问题与答案；长文档不切会超过上下文粒度，且无法定位章节。
- **继续追问：** 两者能否共存？可以，统一为 `Chunk` Schema，以 `source` 区分。
- **陷阱/错误回答：** “所有文本都按 600 字符切”会破坏 QA 单元。
- **代码位置：** `ingestion/representations.py`、`ingestion/pipeline.py`。

## 2. 为什么 Parser、Chunker、Representation、Store 四层解耦？

- **标准回答：** Parser 输出 Section Tree，Chunker 只切结构，Representation 构建检索文本，Store 只处理向量与 Milvus。
- **为什么这样设计：** 变更一种文件格式或切分策略不会耦合到 embedding 与数据库写入。
- **继续追问：** 如何做 Chunking 消融？只替换 `strategy`，其余三层不动。
- **陷阱/错误回答：** 在 Parser 内嵌 embedding 或直接 insert，导致无法单独测试解析与切分。
- **代码位置：** `ingestion/parsers/`、`chunking.py`、`representations.py`、`core/KnowledgeBase.py`。

## 3. Section Tree 对 Retrieval 有什么价值？

- **标准回答：** Tree 提供稳定的 `section_path`、层级和页码，供 summary、过滤、展示和候选解释使用。
- **为什么这样设计：** 同一句正文在不同章节含义不同；章节标题是低成本的语义上下文。
- **继续追问：** overlap 能跨章节吗？不能，否则 metadata 与正文归属不一致。
- **陷阱/错误回答：** 只拍平全文会丢失标题上下文和精确定位能力。
- **代码位置：** `ingestion/tree.py`、`chunking.py`。

## 4. PDF Heading heuristic 的精确实现是什么？

- **标准回答：** 从 `get_text("dict")` 的 span 获取字号，以非表格文本字号中位数为正文；短文本达到正文 1.15 倍时成为标题候选，并按 1.45/1.28/1.15 分级为 H1/H2/H3。
- **为什么这样设计：** 医学指南通常字号层级明显，可在不引入布局模型的前提下恢复足够结构。
- **继续追问：** 标题识别失败怎么办？退化为普通 Section，不能丢全文。
- **陷阱/错误回答：** 把字体规则描述为保证正确的 PDF 语义解析。
- **代码位置：** `ingestion/parsers/pdf.py`。

## 5. 双栏 heuristic 怎么做？

- **标准回答：** 当中位行宽小于页面宽度的 0.62 时，按左栏、右栏、纵向位置排序；否则按纵向、横向位置排序。
- **为什么这样设计：** 修复最常见的两栏阅读顺序，保持 Stage 1 的简单性。
- **继续追问：** 对复杂杂志版式如何处理？记录为已知限制，后续可替换布局模型。
- **陷阱/错误回答：** 声称已经通用支持任意多栏、图文混排。
- **代码位置：** `ingestion/parsers/pdf.py`。

## 6. Table 为什么独立处理？

- **标准回答：** 表格由 `find_tables()` 或 Markdown 管道表转成 `is_table=True` Section；PDF 表格 bbox 从普通正文排除。
- **为什么这样设计：** 表格的行列结构和普通段落不同，重复文本会污染检索。
- **继续追问：** 超长表怎么处理？按行切分，后续块重复表头。
- **陷阱/错误回答：** 直接把表格按 600 字符切，造成表头、行与列失配。
- **代码位置：** `parsers/pdf.py`、`parsers/markdown.py`、`chunking.py`。

## 7. Structural Chunk 的算法过程是什么？

- **标准回答：** 遍历 Section Tree，在当前 Section 内以 `。；！？\n` 分句，累积到约 600 字符，保留 section path/page；短尾可向前合并，连续长文本 hard split。
- **为什么这样设计：** 让每个 Chunk 保持一个明确章节归属，同时避免过大的 dense/BM25 输入。
- **继续追问：** 默认参数？`max_chars=600`、`min_chars=100`、`overlap=80`。
- **陷阱/错误回答：** 将 structural 描述为按 token 或跨章节的全局滑窗。
- **代码位置：** `ingestion/chunking.py`。

## 8. 600/80 是不是最优参数？

- **标准回答：** 不是；它们是固定、可复现的 Stage 1 基线参数。
- **为什么这样设计：** 先固定切分变量，避免检索模型、索引和 Chunk 参数同时变化而不可比较。
- **继续追问：** 何时调优？在固定语料和真实评测集上做独立 Chunking 消融。
- **陷阱/错误回答：** 用当前 Smoke 结果证明 600/80 在医学语料上最优。
- **代码位置：** `chunk_document` 的默认参数。

## 9. 为什么 overlap 不能跨 Section？

- **标准回答：** overlap 只能从同一 Section 的上一块借文本。
- **为什么这样设计：** 跨章节会让 Chunk 的 section path 无法准确说明其正文来源，影响过滤和引用。
- **继续追问：** 如果上一节的语义确实相关？由标题/summary 和后续 reranker 处理，不伪造 Chunk 归属。
- **陷阱/错误回答：** 把跨章节 overlap 当成免费扩展上下文。
- **代码位置：** `_sentence_chunks`、`chunk_document`。

## 10. Fixed、Recursive、Structural 的实验意义？

- **标准回答：** fixed 是无结构滑窗基线；recursive 是按自然分隔符递归的基线；structural 是保留章节边界的默认方案。
- **为什么这样设计：** 三者隔离“结构信息是否有帮助”的变量。
- **继续追问：** recursive 的不变量？非表格块不超过 `max_chars`。
- **陷阱/错误回答：** 把 recursive 当成 structural 的别名，或不验证硬切路径。
- **代码位置：** `chunking.py`、`tests/test_stage1_recursive_chunking.py`。

## 11. summary_dense 与 text_dense 的信息粒度差异？

- **标准回答：** summary_dense 嵌入问题或标题/章节/导语，偏意图；text_dense 嵌入标题、章节与完整正文，偏上下文。
- **为什么这样设计：** 一个查询可能更接近问题表达，也可能依赖正文细节。
- **继续追问：** 当前能比较谁更好？默认真实 embedding 可用于人工 sanity check，但没有 Gold 标注仍不能比较指标优劣。
- **陷阱/错误回答：** 根据 Smoke 的单次排名宣称两个字段的语义效果差异。
- **代码位置：** `representations.py`、`KnowledgeBase.py`。

## 12. Literature summary 为什么不用 LLM？

- **标准回答：** 使用 title、section path 和 lead text 的确定性函数。
- **为什么这样设计：** 可复现、无入库模型成本、不受模型版本影响，便于后续消融。
- **继续追问：** 是否永远不能用 LLM？不是，未来可做独立的 versioned 实验，但不属于 Stage 1。
- **陷阱/错误回答：** 认为 title+lead text 等价于高质量抽象摘要。
- **代码位置：** `make_literature_chunk`、`lead_text`。

## 13. 为什么改成 Milvus-managed BM25？

- **标准回答：** `text` 通过 Milvus Analyzer 和 `FunctionType.BM25` 自动生成 `text_sparse`，删除自管词表和稀疏向量状态。
- **为什么这样设计：** 文档和查询使用同一服务端分析链路，减少 Vocabulary 生命周期和分词不一致问题。
- **继续追问：** 中文分词器？当前 Schema 配置 Jieba。
- **陷阱/错误回答：** 恢复 pkuseg/Vocabulary 并声称与当前 Schema 兼容。
- **代码位置：** `_create_collection`、`_encode_query`。

## 14. BM25 的 TF/IDF/长度归一化原理是什么？

- **标准回答：** BM25 对查询词在文档内的 TF 给予饱和增益，以 IDF 提高稀有词权重，并用文档长度相对平均长度做归一化；常见参数为 k1 与 b。
- **为什么这样设计：** 医学专有词适合精确词法匹配，同时避免长文因重复词无限占优。
- **继续追问：** 本项目在 Python 侧计算吗？不计算，由 Milvus BM25 Function 和稀疏索引负责。
- **陷阱/错误回答：** 把 BM25 当成 dense 向量余弦相似度。
- **代码位置：** `core/KnowledgeBase.py` 的 Function 和索引定义。

## 15. HNSW 的 M、efConstruction、efSearch 是什么？

- **标准回答：** M 是图中每节点的最大邻接连接数；efConstruction 是建图时的候选搜索宽度；efSearch 是查询时的候选搜索宽度。当前固定 M=32、efConstruction=200，查询 dense 路由传 `ef=64`。
- **为什么这样设计：** 固定索引基线，避免 Stage 1 同时调入库和 ANN 参数。
- **继续追问：** M/ef 越大一定越好吗？通常召回可能提高，但内存、建库或查询成本也提高。
- **陷阱/错误回答：** 把 efConstruction 当作每次检索的 top-k。
- **代码位置：** `KnowledgeBase.build_index`、`scripts/03_search_data.py`。

## 16. RRF 为什么不是 Recall？

- **标准回答：** RRF 是把多个排序列表按名次融合的算法；Recall 是需要有标注相关集才能计算的评测指标。
- **为什么这样设计：** Stage 1 只验证混合检索路由能运行，不构造未经验证的效果结论。
- **继续追问：** 如何评估 RRF？在固定 query/相关文档标注上计算 Recall@k、nDCG 等。
- **陷阱/错误回答：** 根据 RRF 返回了结果就报告 Recall 提升。
- **代码位置：** `KnowledgeBase.search`、`scripts/03_search_data.py`。

## 17. 一个 Collection 的好处是什么？

- **标准回答：** QA 与文献共享三路字段、索引、RRF 和过滤表达式，避免跨库合并与分数不可比。
- **为什么这样设计：** Stage 1 的目标是验证多源共存，而不是建立复杂的 corpus registry。
- **继续追问：** 会不会混淆来源？不会，`source`、`source_name` 和文档元数据可过滤和展示。
- **陷阱/错误回答：** 为此阶段引入 source tier、版本表或 MySQL registry。
- **代码位置：** `models.py`、`KnowledgeBase._create_collection`。

## 18. source filter 如何证明多源数据能共存？

- **标准回答：** 同一个 Collection 上分别执行 `source == "qa"` 和 `source == "literature"`，并验证返回结果仅含目标 source。
- **为什么这样设计：** 它同时证明 metadata 入库、Milvus filter 和两类行共存。
- **继续追问：** 是否证明语义质量？不证明，只证明数据与过滤链路正确。
- **陷阱/错误回答：** 将 filter 通过解释成 hybrid 的召回质量提升。
- **代码位置：** `scripts/03_search_data.py`、`smoke_search.json`。

## 19. local_hash 为什么只作为离线回退？

- **标准回答：** 正式默认配置是 `qwen3.7-text-embedding-flash`；local_hash 仅稳定生成 1024 维向量，用于无网络或无密钥时验证 HNSW/COSINE 基础设施。
- **为什么这样设计：** 正式检索使用真实语义 embedding，同时保留无外部依赖的诊断路径。
- **继续追问：** 真实 embedding 的 Smoke 证明什么？证明语义向量、索引与查询链路；没有 Gold 标注仍不证明语义 Recall。
- **陷阱/错误回答：** 用 local_hash 结果声称 dense/hybrid 优于 BM25，或把改 YAML 当成已更新 Milvus 中既有向量。
- **代码位置：** `LocalHashEmbeddings`、`app_config.yaml`、`02_ingest_data.py --recreate`。

## 20. Stage 1 为什么不直接加入 Reranker？

- **标准回答：** Reranker 需要稳定的 candidate recall、标注或评测口径；先冻结多源入库与候选召回基线。
- **为什么这样设计：** 否则无法区分提升来自 Parser、Chunking、embedding、BM25、RRF 还是 reranker。
- **继续追问：** Stage 1 有什么可交付？可复现的候选集合和 metadata。
- **陷阱/错误回答：** 为了看起来生产化提前加入 cross-encoder、Citation 或安全链路。
- **代码位置：** `KnowledgeBase.search`、`docs/stage1_ingestion_implementation.md`。

## 21. 下一阶段 Candidate Recall / Rerank 如何连接？

- **标准回答：** 先由现有 dense、BM25 或 hybrid 请求产生 top-k `Document`，再以 query 与 `text` 为输入对候选重打分，输出重排列表。
- **为什么这样设计：** 保留 Stage 1 的 `pk`、source、doc_id、section_path、page，可追踪候选变化而无需重建语料。
- **继续追问：** Reranker 是否写入 Milvus？Stage 2 的在线重排不要求改变当前 Schema。
- **陷阱/错误回答：** 用 reranker 替代解析/切分，或在无评测时宣称最终效果。
- **代码位置：** `core/KnowledgeBase.py` 的 `search()` 返回值与 `Chunk` 元数据。
