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
from app.providers.serpapi_product import SerpApiProductProvider
from app.providers.serper_web_search import SerperWebSearchProvider
from app.providers.shipping import RateTableShippingProvider


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
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        )
        if settings.product_provider == "placeholder":
            product: ProductSearchProvider = PlaceholderProductSearchProvider()
        elif settings.product_provider == "serpapi":
            product = SerpApiProductProvider(api_key=os.getenv("SERPAPI_API_KEY", ""))
        else:
            product = HttpProductSearchProvider(
                api_url=settings.product_api_url or "",
                api_key=os.getenv("OMNIMATCH_PRODUCT_API_KEY", ""),
            )
        if settings.web_search_provider == "placeholder":
            web_search: WebSearchProvider = PlaceholderWebSearchProvider()
        elif settings.web_search_provider == "serper":
            web_search = SerperWebSearchProvider(api_key=os.getenv("SERPER_API_KEY", ""))
        else:
            web_search = HttpWebSearchProvider(
                api_url=settings.web_search_api_url or "",
                api_key=os.getenv("OMNIMATCH_WEB_SEARCH_API_KEY", ""),
            )
        if settings.shipping_provider == "placeholder":
            shipping: ShippingProvider = PlaceholderShippingProvider()
        elif settings.shipping_provider == "rate_table":
            shipping = RateTableShippingProvider()
        else:
            shipping = RateTableShippingProvider()
        return cls(
            llm=llm,
            product=product,
            web_search=web_search,
            shipping=shipping,
        )
