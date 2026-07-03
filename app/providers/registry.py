from __future__ import annotations

import os
from dataclasses import dataclass

from app.config import OmniMatchSettings
from app.providers.base import LLMProvider, ProductSearchProvider, ShippingProvider, WebSearchProvider
from app.providers.http_product import HttpProductSearchProvider
from app.providers.http_web_search import HttpWebSearchProvider
from app.providers.openai_llm import OpenAILLMProvider
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
        llm: LLMProvider = (
            PlaceholderLLMProvider()
            if settings.llm_provider == "placeholder"
            else OpenAILLMProvider(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=settings.llm_model,
            )
        )
        product: ProductSearchProvider = (
            PlaceholderProductSearchProvider()
            if settings.product_provider == "placeholder"
            else HttpProductSearchProvider(
                api_url=settings.product_api_url or "",
                api_key=os.getenv("OMNIMATCH_PRODUCT_API_KEY", ""),
            )
        )
        web_search: WebSearchProvider = (
            PlaceholderWebSearchProvider()
            if settings.web_search_provider == "placeholder"
            else HttpWebSearchProvider(
                api_url=settings.web_search_api_url or "",
                api_key=os.getenv("OMNIMATCH_WEB_SEARCH_API_KEY", ""),
            )
        )
        return cls(
            llm=llm,
            product=product,
            web_search=web_search,
            shipping=PlaceholderShippingProvider(),
        )
