from __future__ import annotations

import time
from typing import Any

import httpx

from app.providers.base import ProviderError, ProviderResult


class SerperWebSearchProvider:
    provider = "serper"
    api_url = "https://google.serper.dev/search"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=20)

    async def search(self, query: str) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        response = await self.client.post(
            self.api_url,
            json={"q": query},
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
        )
        if response.status_code >= 400:
            detail = self._error_detail(response)
            raise ProviderError(
                self.provider,
                f"serper web search returned {response.status_code}: {detail}",
            )
        payload = response.json()
        results = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("link", "")),
                "snippet": str(item.get("snippet", "")),
            }
            for item in payload.get("organic", [])
        ]
        return ProviderResult(
            provider=self.provider,
            provider_mode="real",
            latency_ms=max(0, int((time.perf_counter() - start) * 1000)),
            data=results,
            warnings=[],
            response_summary=f"normalized {len(results)} serper organic results",
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            text = response.text.strip()
            return text[:200] if text else "no response body"
        if isinstance(payload, dict):
            for key in ("message", "error", "detail"):
                value = payload.get(key)
                if value:
                    return str(value)
        return response.text.strip()[:200] or "no response body"
