import os
from typing import List

from tavily import TavilyClient
from langchain_core.documents import Document


def tavily_search(
    query: str,
    cnt: int = 5
) -> List[Document]:

    client = TavilyClient(
        api_key=os.environ["TAVILY_API_KEY"]
    )

    # ① Search：先找网页
    search_response = client.search(
        query=query,
        search_depth="basic",
        max_results=min(cnt, 20),
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