"""Stage 1 Milvus 存储：两个 dense 字段与 Milvus 托管 BM25。"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker, WeightedRanker

from ..config.models import AppConfig, SearchRequest, SingleSearchRequest
from ..ingestion.models import Chunk
from ..ingestion.representations import fit_varchar
from .insert import insert_rows
from .utils import create_embedding_client

ALL_FIELDS = [
    "pk", "text", "summary", "document", "source", "source_name", "doc_id", "chunk_id",
    "department", "title", "section_path", "page", "summary_dense", "text_dense", "text_sparse",
]


class LocalHashEmbeddings(Embeddings):
    """为已提供的 Stage 1 语料生成离线、确定性的 1024 维 embedding。"""
    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        compact = " ".join((text or "").lower().split())
        grams = [compact[index:index + size] for size in (1, 2, 3) for index in range(max(0, len(compact) - size + 1))]
        for gram in grams or [" "]:
            value = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
            vector[value % self.dimension] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class MedicalHybridKnowledgeBase:
    """使用 Milvus 托管稀疏检索的当前核心 Schema。"""
    def __init__(self, app_config: AppConfig):
        self.milvus_config = app_config.milvus
        self.embedding_config = app_config.embedding
        if self.embedding_config.summary_dense.dimension != self.embedding_config.text_dense.dimension:
            raise ValueError("summary_dense and text_dense dimensions must be equal")
        self.summary_embedding = self._make_embedding(self.embedding_config.summary_dense)
        self.text_embedding = self._make_embedding(self.embedding_config.text_dense)
        self.client = MilvusClient(uri=self.milvus_config.uri, token=self.milvus_config.token)

    @staticmethod
    def _make_embedding(config) -> Embeddings:
        if config.provider == "local_hash":
            return LocalHashEmbeddings(config.dimension)
        return create_embedding_client(config)

    def _create_collection(self, recreate: bool = False) -> None:
        name = self.milvus_config.collection_name
        exists = self.client.has_collection(collection_name=name)
        if exists and not recreate:
            return
        if exists:
            self.client.drop_collection(collection_name=name)
        dim = self.embedding_config.summary_dense.dimension
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True,
                         enable_match=True, analyzer_params={"tokenizer": "jieba"})
        for field in ("summary", "document"):
            schema.add_field(field_name=field, datatype=DataType.VARCHAR, max_length=65535)
        for field in ("source", "source_name", "doc_id", "department", "title", "section_path"):
            schema.add_field(field_name=field, datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64)
        schema.add_field(field_name="page", datatype=DataType.INT64)
        schema.add_field(field_name="summary_dense", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(name="bm25_text_to_sparse", function_type=FunctionType.BM25,
                                     input_field_names=["text"], output_field_names=["text_sparse"]))
        self.client.create_collection(collection_name=name, schema=schema)

    def build_index(self) -> None:
        params = self.client.prepare_index_params()
        for field in ("summary_dense", "text_dense"):
            params.add_index(field_name=field, index_type="HNSW", index_name=f"{field}_index",
                             metric_type="COSINE", params={"M": 32, "efConstruction": 200})
        params.add_index(field_name="text_sparse", index_type="SPARSE_INVERTED_INDEX", index_name="text_sparse_index",
                         metric_type="BM25", params={"inverted_index_algo": "DAAT_MAXSCORE"})
        self.client.create_index(collection_name=self.milvus_config.collection_name, index_params=params)
        self.client.load_collection(self.milvus_config.collection_name)

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        rows: list[dict] = []
        for chunk in chunks:
            row = chunk.model_dump()
            row["page"] = int(row["page"] or 0)
            for field, limit in (("pk", 128), ("text", 65535), ("summary", 65535), ("document", 65535),
                                 ("source", 4096), ("source_name", 4096), ("doc_id", 4096),
                                 ("department", 4096), ("title", 4096), ("section_path", 4096)):
                row[field] = fit_varchar(str(row[field] or ""), limit)
            row["summary_dense"] = self.summary_embedding.embed_documents([row["summary"]])[0]
            row["text_dense"] = self.text_embedding.embed_documents([row["text"]])[0]
            rows.append(row)
        if rows:
            insert_rows(self.client, self.milvus_config.collection_name, rows, show_progress=True)
        return len(rows)

    def _encode_query(self, query: str, anns_field: str):
        if anns_field == "summary_dense":
            return self.summary_embedding.embed_query(query)
        if anns_field == "text_dense":
            return self.text_embedding.embed_query(query)
        if anns_field == "text_sparse":
            return query  # Milvus BM25 owns sparse query analysis and encoding.
        raise ValueError(f"Unsupported ANN field: {anns_field}")

    def _single_search(self, query: str, request: SingleSearchRequest, collection: str, fields: list[str]):
        return self.client.search(collection_name=collection, data=[self._encode_query(query, request.anns_field)],
                                  anns_field=request.anns_field, filter=request.expr or "", limit=request.limit,
                                  output_fields=fields,
                                  search_params={"metric_type": request.metric_type, "params": request.search_params})[0]

    def search(self, req: SearchRequest) -> list[Document]:
        if len(req.requests) == 1:
            hits = self._single_search(req.query, req.requests[0], req.collection_name, req.output_fields)
        else:
            requests = [AnnSearchRequest(data=[self._encode_query(req.query, item.anns_field)], anns_field=item.anns_field,
                param={"metric_type": item.metric_type, "params": item.search_params}, limit=item.limit,
                expr=item.expr or "") for item in req.requests]
            ranker = RRFRanker(req.fuse.k) if req.fuse and req.fuse.method == "rrf" else WeightedRanker(*(req.fuse.weights if req.fuse else [1.0] * len(requests)))
            hits = self.client.hybrid_search(collection_name=req.collection_name, reqs=requests, ranker=ranker,
                                              limit=req.limit, output_fields=req.output_fields)[0]
        documents: list[Document] = []
        for hit in hits:
            # pymilvus 将请求的输出字段放在 ``entity`` 中，不能把有效
            # Hit 误判为空文档。
            entity = hit.get("entity", hit)
            metadata = {field: entity.get(field, "") for field in ALL_FIELDS if field in entity}
            metadata["pk"] = metadata.get("pk") or hit.get("id", "")
            metadata["distance"] = hit.get("distance", 0.0)
            documents.append(Document(page_content=entity.get("text", ""), metadata=metadata))
        return documents
