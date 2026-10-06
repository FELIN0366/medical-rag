import os
from typing import List

from tavily import TavilyClient
from langchain_core.documents import Document


REQUEST_TIMEOUT_SECONDS = 30


def tavily_search(
    query: str,
    cnt: int = 5
) -> List[Document]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # 由 AgentTools 转成受控 JSON，避免 ToolNode 返回非 JSON 错误文本。
        raise KeyError("TAVILY_API_KEY")

    client = TavilyClient(
        api_key=api_key
    )

    # ① Search：先找网页
    search_response = client.search(
        query=query,
        search_depth="basic",
        max_results=min(cnt, 20),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    urls = [
        item["url"]
        for item in search_response.get("results", [])
        if item.get("url")
    ]

    if not urls:
        return []

    # ② Extract：再取网页真正内容
    extract_response = client.extract(
        urls=urls,
        query=query,
        chunks_per_source=3,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    docs = []

    # ③ 转成 LangChain Document
    for item in extract_response.get("results", []):

        raw_content = item.get("raw_content", "")

        if not raw_content:
            continue

        docs.append(
            Document(
                page_content=raw_content,
                metadata={
                    "url": item.get("url", ""),
                    "source": "tavily_web",
                }
            )
        )

    return docs

if __name__ == "__main__":

    docs = tavily_search(
        query="西瓜的药用",
        cnt=10,
    )

    for i, doc in enumerate(docs):
        print(f"\n===== {i + 1} =====")
        print(doc.page_content)
        print(doc.metadata)
