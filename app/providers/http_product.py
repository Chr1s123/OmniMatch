from __future__ import annotations

import time
from typing import Any

import httpx

from app.providers.base import ProviderError, ProviderResult


class HttpProductSearchProvider:
    provider = "http_product"

    def __init__(
        self,
        api_url: str,
        api_key: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=20)

    async def search(self, query: str, platforms: list[str]) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        response = await self.client.get(
            self.api_url,
            params={"q": query, "platforms": ",".join(platforms)},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if response.status_code >= 400:
            detail = self._error_detail(response)
            raise ProviderError(
                self.provider,
                f"product provider returned {response.status_code}: {detail}",
            )
        payload = response.json()
        items = [self._normalize_item(item) for item in payload.get("items", [])]
        return ProviderResult(
            provider=self.provider,
            provider_mode="real",
            latency_ms=max(0, int((time.perf_counter() - start) * 1000)),
            data=items,
            warnings=[],
            response_summary=f"normalized {len(items)} product items",
        )

    @staticmethod
    def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item["id"]),
            "platform": str(item.get("platform", "unknown")),
            "title": str(item["title"]),
            "price": float(item["price"]),
            "currency": str(item.get("currency", "CNY")),
            "url": str(item["url"]),
            "rating": float(item.get("rating", 0)),
            "evidence": item.get("evidence", []),
            "material": item.get("material"),
        }

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            text = response.text.strip()
            return text[:200] if text else "no response body"
        if isinstance(payload, dict):
            for key in ("error", "message", "detail"):
                value = payload.get(key)
                if value:
                    return str(value)
        return response.text.strip()[:200] or "no response body"
