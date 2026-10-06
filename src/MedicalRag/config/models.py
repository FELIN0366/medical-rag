"""Stage 1 核心 Schema 的配置与检索请求模型。"""
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


class RerankerConfig(BaseModel):
    """Stage 2 DashScope 重排服务配置。"""
    provider: Literal["dashscope"] = "dashscope"
    model: str = "qwen3.7-text-rerank"
    api_key_env: str = "DASHSCOPE_API_KEY"
    workspace_id_env: str = "DASHSCOPE_WORKSPACE_ID"
    region: str = "cn-beijing"
    top_n: int = Field(default=30, ge=1, le=30)
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=5)
    instruct: str = (
        "Given a medical question, rank passages by whether they contain evidence "
        "that directly answers the question. Prefer directly supporting passages "
        "over passages that are only topically related."
    )


class RetrievalStage2Config(BaseModel):
    """Stage 2 固定的三路召回与重排参数。"""
    per_route_k: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=200)
    candidate_k: int = Field(default=30, ge=1, le=100)
    final_k: int = Field(default=5, ge=1, le=50)

class LLMConfig(BaseModel):
    provider: Literal["openai", "ollama"] = "ollama"
    model: str = "qwen3:32b"
    base_url: Optional[str] = None
    env_key_name: Optional[str] = None
    proxy: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: float = Field(default=30.0, gt=0)
    # 远程 Agent 节点发生超时后由图状态降级，不在 SDK 内部重复阻塞同一请求。
    max_retries: int = Field(default=0, ge=0, le=5)

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
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval_stage2: RetrievalStage2Config = Field(default_factory=RetrievalStage2Config)
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
    # 主键是评测 Gold、重排审计与下游引用候选的稳定标识，不能依赖 Milvus 命中对象的隐式 id。
    output_fields: list[OutputFields] = Field(default_factory=lambda: ["pk", "text", "summary", "document"])
    fuse: Optional[FusionSpec] = Field(default_factory=FusionSpec)
    limit: int = Field(default=5, gt=0, le=500)
