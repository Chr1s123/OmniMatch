from __future__ import annotations

from dataclasses import dataclass

from app.config import OmniMatchSettings
from app.providers.base import LLMProvider, ProductSearchProvider, ShippingProvider, WebSearchProvider
from app.providers.placeholder import (
    PlaceholderLLMProvider,
    PlaceholderProductSearchProvider,
    PlaceholderShippingProvider,
    PlaceholderWebSearchProvider,
)


@dataclass(frozen=True)
class ProviderRegistry:
    llm: LLMProvider
    product: ProductSearchProvider
    web_search: WebSearchProvider
    shipping: ShippingProvider

    @classmethod
    def from_settings(cls, settings: OmniMatchSettings) -> "ProviderRegistry":
        return cls(
            llm=PlaceholderLLMProvider(),
            product=PlaceholderProductSearchProvider(),
            web_search=PlaceholderWebSearchProvider(),
            shipping=PlaceholderShippingProvider(),
        )
