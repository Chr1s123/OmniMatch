from __future__ import annotations

import time
from typing import Any

from app.providers.base import ProviderResult


def _latency_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


class PlaceholderLLMProvider:
    provider = "placeholder_llm"

    async def plan_next_action(self, messages: list[dict[str, Any]]) -> ProviderResult[dict[str, Any]]:
        start = time.perf_counter()
        return ProviderResult(
            provider=self.provider,
            provider_mode="placeholder",
            latency_ms=_latency_ms(start),
            data={"action": "finish_if_enough_else_search", "arguments": {}},
            warnings=["placeholder LLM used"],
            response_summary="deterministic placeholder plan",
        )


class PlaceholderProductSearchProvider:
    provider = "placeholder_product"

    async def search(self, query: str, platforms: list[str]) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        data: list[dict[str, Any]] = []
        for platform in platforms:
            slug = platform.lower().replace(" ", "-")
            data.append(
                {
                    "id": f"{slug}-canvas-travel-set",
                    "platform": platform,
                    "title": f"{platform} canvas travel set",
                    "price": 198,
                    "currency": "CNY",
                    "rating": 4.6,
                    "url": f"https://example.com/{slug}/canvas-travel-set",
                    "evidence": ["placeholder catalog fixture"],
                    "material": "canvas",
                }
            )
        return ProviderResult(
            provider=self.provider,
            provider_mode="placeholder",
            latency_ms=_latency_ms(start),
            data=data,
            warnings=["placeholder product data used"],
            response_summary=f"placeholder products for query={query!r}",
        )


class PlaceholderWebSearchProvider:
    provider = "placeholder_web_search"

    async def search(self, query: str) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        return ProviderResult(
            provider=self.provider,
            provider_mode="placeholder",
            latency_ms=_latency_ms(start),
            data=[{"title": "Travel set buying guide", "url": "https://example.com/guide"}],
            warnings=["placeholder web evidence used"],
            response_summary=f"placeholder web result for query={query!r}",
        )


class PlaceholderShippingProvider:
    provider = "placeholder_shipping"

    async def estimate(
        self,
        product: dict[str, Any],
        destination: str | None,
    ) -> ProviderResult[dict[str, Any]]:
        start = time.perf_counter()
        return ProviderResult(
            provider=self.provider,
            provider_mode="placeholder",
            latency_ms=_latency_ms(start),
            data={"shipping": 20, "tax": 0, "destination": destination},
            warnings=[],
            response_summary="placeholder shipping estimate",
        )
