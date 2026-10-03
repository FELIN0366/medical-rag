"""
RAG基础评测
"""
import logging
import argparse
from MedicalRag.config.loader import ConfigLoader
from MedicalRag.config.models import SearchRequest, SingleSearchRequest
from MedicalRag.rag.SimpleRag import SimpleRAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
import os
from langchain_community.embeddings import DashScopeEmbeddings
from MedicalRag.rag.RagEvaluate import RagasRagEvaluate
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-retrieval",
        action="store_true",
        help="仅使用 summary_dense 单路检索进行评测",
    )
    args = parser.parse_args()

    # 加载配置
    config_manager = ConfigLoader()
    search_config = None
    if args.single_retrieval:
        search_config = SearchRequest(
            collection_name=config_manager.config.milvus.collection_name,
            requests=[SingleSearchRequest(anns_field="summary_dense")],
            output_fields=["summary", "document", "source", "source_name", "doc_id", "chunk_id", "department", "title", "section_path", "page", "text"],
            limit=10,
        )
        logger.info("单路检索评测已启用：summary_dense")

    # 创建基础RAG系统
    rag = SimpleRAG(config_manager.config, search_config=search_config)
    eval_data = load_dataset("json", data_files="data/eval/new_qa_200.jsonl", split="train")
    llm_config = config_manager.config.llm
    qwen_llm = ChatOpenAI(
        base_url=llm_config.base_url,
        model=llm_config.model,
        api_key=os.getenv(llm_config.env_key_name),
        temperature=0.0,
        extra_body={
            "enable_thinking": False
        }
    )
    embedding_config = config_manager.config.embedding
    qwen_embedding = DashScopeEmbeddings(
        model=embedding_config.summary_dense.model,
        dashscope_api_key=os.getenv(embedding_config.summary_dense.env_key_name)
    )
    eval = RagasRagEvaluate(rag_components=rag, eval_datasets=eval_data, eval_llm=qwen_llm, embedding=qwen_embedding)
    eval.do_sample(10)  # 根据需要进行快速修改
    print(eval.do_evaluate(datasets_query_field_name="new_question", datasets_reference_field_name="answer"))

if __name__ == "__main__":
    main()
