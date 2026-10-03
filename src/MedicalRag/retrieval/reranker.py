"""qwen3.7-text-rerank 的 DashScope Model Studio 客户端。"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config.models import RerankerConfig


@dataclass(frozen=True)
class RerankResult:
    """单次 Rerank 请求内的候选索引与相对相关性分数。"""
    index: int
    score: float


class DashScopeReranker:
    """一个 query 对整个候选列表发起一次异步 DashScope 重排请求。"""
    def __init__(self, config: RerankerConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.client = client

    def _url(self) -> str:
        workspace_id = os.getenv(self.config.workspace_id_env)
        if not workspace_id:
            raise RuntimeError(f"未配置 Reranker 所需环境变量：{self.config.workspace_id_env}")
        return f"https://{workspace_id}.{self.config.region}.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def _headers(self) -> dict[str, str]:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"未配置 Reranker 所需环境变量：{self.config.api_key_env}")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        if not documents:
            return []
        if len(documents) > self.config.top_n:
            raise ValueError(f"Reranker 候选数不得超过 {self.config.top_n}")
        requested_top_n = min(top_n or self.config.top_n, len(documents))
        payload = {
            "model": self.config.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": requested_top_n, "instruct": self.config.instruct},
        }
        response = await self._post_with_retry(self._url(), payload)
        data = response.json()
        try:
            raw_results = data["output"]["results"]
            results = [RerankResult(index=int(item["index"]), score=float(item["relevance_score"]))
                       for item in raw_results]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Reranker 响应结构无效：{data}") from exc
        if any(item.index < 0 or item.index >= len(documents) for item in results):
            raise RuntimeError("Reranker 响应包含超出候选范围的 index")
        return results

    async def _post_with_retry(self, url: str, payload: dict) -> httpx.Response:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.config.timeout)
        try:
            retrying = AsyncRetrying(
                retry=retry_if_exception_type(httpx.HTTPError),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                stop=stop_after_attempt(self.config.max_retries + 1),
                reraise=True,
            )
            async for attempt in retrying:
                with attempt:
                    response = await client.post(url, headers=self._headers(), json=payload)
                    response.raise_for_status()
                    return response
            raise RuntimeError("Reranker 重试流程未返回响应")
        finally:
            if owns_client:
                await client.aclose()
