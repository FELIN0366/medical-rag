"""Configuration and retrieval request models for the Stage 1 core schema."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

class MilvusConfig(BaseModel):
    uri: str = "http://localhost:19530"
    token: Optional[str] = None
    collection_name: str = "medical_knowledge"
    drop_old: bool = False

class DenseConfig(BaseModel):
    provider: Literal["openai", "ollama", "local_hash"] = "local_hash"
    model: str = "deterministic-hash-1024"
    base_url: Optional[str] = None
    env_key_name: Optional[str] = None
    proxy: Optional[str] = None
    dimension: int = 1024

class SparseConfig(BaseModel):
    provider: Literal["Milvus"] = "Milvus"

class EmbeddingConfig(BaseModel):
    summary_dense: DenseConfig = Field(default_factory=DenseConfig)
    text_dense: DenseConfig = Field(default_factory=DenseConfig)
    text_sparse: SparseConfig = Field(default_factory=SparseConfig)

class LLMConfig(BaseModel):
    provider: Literal["openai", "ollama"] = "ollama"
    model: str = "qwen3:32b"
    base_url: Optional[str] = None
    env_key_name: Optional[str] = None
    proxy: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None

class DataConfig(BaseModel):
    default_source: str = "qa"
    default_source_name: str = "huatuo_qa"

class MultiDialogueRagConfig(BaseModel):
    estimate_token_fun: str = "tiktoken"
    llm_max_token: int = 102400
    max_token_threshold: float = 1.01
    cut_dialogue_scale: int = Field(default=2, ge=2)
    smith_debug: bool = False
    console_debug: bool = True
    thinking_in_context: bool = False

class AgentConfig(BaseModel):
    mode: Literal["analysis", "fast", "normal"] = "analysis"
    max_attempts: int = 2
    network_search_enabled: bool = True
    network_search_cnt: int = 10
    auto_search_param: bool = True

class AppConfig(BaseModel):
    milvus: MilvusConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    multi_dialogue_rag: MultiDialogueRagConfig = Field(default_factory=MultiDialogueRagConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

AnnsField = Literal["summary_dense", "text_dense", "text_sparse"]
OutputFields = Literal["pk", "text", "summary", "document", "source", "source_name", "doc_id", "chunk_id", "department", "title", "section_path", "page"]

class FusionSpec(BaseModel):
    method: Literal["rrf", "weighted"] = "rrf"
    k: int = Field(default=60, gt=0, le=200)
    weights: list[float] = Field(default_factory=lambda: [0.5, 0.5])

class SingleSearchRequest(BaseModel):
    anns_field: AnnsField = "summary_dense"
    metric_type: Literal["COSINE", "IP", "BM25"] = "COSINE"
    search_params: dict = Field(default_factory=lambda: {"ef": 64})
    limit: int = Field(default=50, gt=0, le=500)
    expr: str = ""

class SearchRequest(BaseModel):
    query: str = ""
    collection_name: str = "medical_knowledge"
    requests: list[SingleSearchRequest] = Field(default_factory=lambda: [SingleSearchRequest()])
    output_fields: list[OutputFields] = Field(default_factory=lambda: ["text", "summary", "document"])
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec)
    limit: int = Field(default=5, gt=0, le=10)
