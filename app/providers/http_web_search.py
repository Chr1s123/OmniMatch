from __future__ import annotations

import time
from typing import Any

import httpx

from app.providers.base import ProviderError, ProviderResult


class HttpWebSearchProvider:
    provider = "http_web_search"

    def __init__(
        self,
        api_url: str,
        api_key: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=20)

    async def search(self, query: str) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        response = await self.client.get(
            self.api_url,
            params={"q": query},
            headers={"x-api-key": self.api_key},
        )
        if response.status_code >= 400:
            raise ProviderError(self.provider, f"web search provider returned {response.status_code}")
        payload = response.json()
        results = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("snippet", "")),
            }
            for item in payload.get("results", [])
        ]
        return ProviderResult(
            provider=self.provider,
            provider_mode="real",
            latency_ms=max(0, int((time.perf_counter() - start) * 1000)),
            data=results,
            warnings=[],
            response_summary=f"normalized {len(results)} web results",
        )
