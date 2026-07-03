from __future__ import annotations

import time
from typing import Any

from app.providers.base import ProviderResult


class RateTableShippingProvider:
    provider = "rate_table"

    async def estimate(
        self,
        product: dict[str, Any],
        destination: str | None,
    ) -> ProviderResult[dict[str, Any]]:
        start = time.perf_counter()
        price = float(product.get("price") or 0)
        platform = str(product.get("platform") or "").lower()

        if price >= 299:
            shipping = 0
        elif "amazon" in platform:
            shipping = 25
        elif "ebay" in platform:
            shipping = 30
        elif "aliexpress" in platform:
            shipping = 15
        else:
            shipping = 20

        return ProviderResult(
            provider=self.provider,
            provider_mode="real",
            latency_ms=max(0, int((time.perf_counter() - start) * 1000)),
            data={"shipping": shipping, "tax": 0, "destination": destination},
            warnings=[],
            response_summary="rate table shipping estimate",
        )
