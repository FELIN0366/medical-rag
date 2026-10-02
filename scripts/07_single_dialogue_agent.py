from MedicalRag.agent.SearchGraph import SearchGraph
import logging
from MedicalRag.agent.tools import tencent_cloud_search,tavily_search
from MedicalRag.config.loader import ConfigLoader
from langchain_openai import ChatOpenAI
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

if __name__ == "__main__":
    config_manager = ConfigLoader()
    # config_manager.change({
    #     "llm.model":"qwen3:32b",
    #     "agent.network_search_cnt": 5
    # })
    llm_config = config_manager.config.llm
    power_model = ChatOpenAI(
        base_url=llm_config.base_url,
        model=llm_config.model,
        api_key=os.getenv(llm_config.env_key_name),
        temperature=0.1
    )
    graph = SearchGraph(config_manager.config, power_model=power_model, websearch_func=tavily_search)
    result = graph.answer("腹部疼痛的临床诊断")
    print(result)
