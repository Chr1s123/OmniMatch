from __future__ import annotations

import time
from typing import Any

import httpx

from app.providers.base import ProviderError, ProviderResult


class SerpApiProductProvider:
    provider = "serpapi_product"
    api_url = "https://serpapi.com/search"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=20)

    async def search(self, query: str, platforms: list[str]) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        response = await self.client.get(
            self.api_url,
            params={
                "engine": "google_shopping",
                "q": query,
                "api_key": self.api_key,
            },
        )
        if response.status_code >= 400:
            detail = self._error_detail(response)
            raise ProviderError(
                self.provider,
                f"serpapi product provider returned {response.status_code}: {detail}",
            )
        payload = response.json()
        raw_items = payload.get("shopping_results", [])
        matched_items = [
            item for item in raw_items if self._matches_platform(item, platforms)
        ]
        warnings: list[str] = []
        selected_items = matched_items
        if raw_items and platforms and not matched_items:
            selected_items = raw_items
            warnings.append("platform filter matched 0; used unfiltered SerpApi results")
        items = [self._normalize_item(item) for item in selected_items]
        return ProviderResult(
            provider=self.provider,
            provider_mode="real",
            latency_ms=max(0, int((time.perf_counter() - start) * 1000)),
            data=items,
            warnings=warnings,
            response_summary=f"normalized {len(items)} serpapi shopping results",
        )

    @staticmethod
    def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        evidence = [
            str(value)
            for value in (item.get("delivery"), item.get("snippet"))
            if value
        ]
        return {
            "id": str(item.get("product_id") or item.get("position") or item["title"]),
            "platform": str(item.get("source", "Google Shopping")),
            "title": str(item["title"]),
            "price": float(item.get("extracted_price") or 0),
            "currency": str(item.get("currency") or SerpApiProductProvider._currency_from_price(item)),
            "url": str(item.get("link") or item.get("product_link") or ""),
            "rating": float(item.get("rating") or 0),
            "evidence": evidence,
            "material": None,
        }

    @staticmethod
    def _matches_platform(item: dict[str, Any], platforms: list[str]) -> bool:
        if not platforms:
            return True
        source = str(item.get("source", "")).lower()
        return any(platform.lower() in source for platform in platforms)

    @staticmethod
    def _currency_from_price(item: dict[str, Any]) -> str:
        price = str(item.get("price", ""))
        if price.startswith("$"):
            return "USD"
        if price.startswith("¥") or price.startswith("￥"):
            return "CNY"
        if price.startswith("€"):
            return "EUR"
        if price.startswith("£"):
            return "GBP"
        return "CNY"

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
