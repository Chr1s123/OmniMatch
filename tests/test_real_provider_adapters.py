import httpx
import pytest

from app.config import OmniMatchSettings
from app.providers.base import ProviderError
from app.providers.http_product import HttpProductSearchProvider
from app.providers.http_web_search import HttpWebSearchProvider
from app.providers.openai_llm import OpenAILLMProvider
from app.providers.registry import ProviderRegistry
from app.providers.serpapi_product import SerpApiProductProvider
from app.providers.serper_web_search import SerperWebSearchProvider
from app.providers.shipping import RateTableShippingProvider


@pytest.mark.asyncio
async def test_http_product_provider_normalizes_items():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer unit-key"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "raw-1",
                        "platform": "Amazon",
                        "title": "Canvas travel set",
                        "price": 199,
                        "currency": "CNY",
                        "url": "https://example.com/raw-1",
                        "rating": 4.7,
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HttpProductSearchProvider(
        api_url="https://product.example/search",
        api_key="unit-key",
        client=client,
    )

    result = await provider.search("旅行三件套", platforms=["Amazon"])

    assert result.provider_mode == "real"
    assert result.data[0]["id"] == "raw-1"
    assert result.data[0]["platform"] == "Amazon"


@pytest.mark.asyncio
async def test_http_product_provider_includes_error_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_api_key"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HttpProductSearchProvider(
        api_url="https://product.example/search",
        api_key="unit-key",
        client=client,
    )

    with pytest.raises(ProviderError, match=r"invalid_api_key"):
        await provider.search("旅行三件套", platforms=["Amazon"])


@pytest.mark.asyncio
async def test_serpapi_product_provider_normalizes_google_shopping_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["engine"] == "google_shopping"
        assert request.url.params["q"] == "旅行三件套 小众"
        assert request.url.params["api_key"] == "serpapi-key"
        return httpx.Response(
            200,
            json={
                "shopping_results": [
                    {
                        "product_id": "shop-1",
                        "source": "Amazon",
                        "title": "Canvas Travel Organizer Set",
                        "extracted_price": 199.99,
                        "price": "$199.99",
                        "currency": "USD",
                        "link": "https://example.com/shop-1",
                        "rating": 4.6,
                        "delivery": "Free delivery",
                        "snippet": "Canvas packing cubes for travel.",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SerpApiProductProvider(api_key="serpapi-key", client=client)

    result = await provider.search("旅行三件套 小众", platforms=["Amazon"])

    assert result.provider == "serpapi_product"
    assert result.provider_mode == "real"
    assert result.data == [
        {
            "id": "shop-1",
            "platform": "Amazon",
            "title": "Canvas Travel Organizer Set",
            "price": 199.99,
            "currency": "USD",
            "url": "https://example.com/shop-1",
            "rating": 4.6,
            "evidence": ["Free delivery", "Canvas packing cubes for travel."],
            "material": None,
        }
    ]


@pytest.mark.asyncio
async def test_serpapi_product_provider_falls_back_when_platform_filter_matches_nothing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "shopping_results": [
                    {
                        "product_id": "down-1",
                        "source": "Patagonia Burlington",
                        "title": "Patagonia Down Jacket",
                        "extracted_price": 288,
                        "price": "$288",
                        "link": "https://example.com/down-1",
                        "rating": 4.8,
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SerpApiProductProvider(api_key="serpapi-key", client=client)

    result = await provider.search("羽绒服", platforms=["Amazon", "eBay", "AliExpress", "Shopee"])

    assert result.data[0]["id"] == "down-1"
    assert result.data[0]["platform"] == "Patagonia Burlington"
    assert result.warnings == ["platform filter matched 0; used unfiltered SerpApi results"]


@pytest.mark.asyncio
async def test_serpapi_product_provider_includes_error_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Invalid API key"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SerpApiProductProvider(api_key="bad-key", client=client)

    with pytest.raises(ProviderError, match=r"Invalid API key"):
        await provider.search("旅行三件套", platforms=["Amazon"])


@pytest.mark.asyncio
async def test_http_web_search_provider_normalizes_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Guide", "url": "https://example.com/guide", "snippet": "ok"}
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HttpWebSearchProvider(
        api_url="https://search.example/query",
        api_key="unit-key",
        client=client,
    )

    result = await provider.search("travel set material")

    assert result.provider_mode == "real"
    assert result.data[0]["title"] == "Guide"


@pytest.mark.asyncio
async def test_serper_web_search_provider_posts_query_and_normalizes_organic_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "https://google.serper.dev/search"
        assert request.headers["x-api-key"] == "serper-key"
        assert request.headers["content-type"] == "application/json"
        assert request.read() == b'{"q":"travel set material"}'
        return httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Material guide",
                        "link": "https://example.com/guide",
                        "snippet": "Canvas and nylon comparison",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SerperWebSearchProvider(api_key="serper-key", client=client)

    result = await provider.search("travel set material")

    assert result.provider == "serper"
    assert result.provider_mode == "real"
    assert result.data == [
        {
            "title": "Material guide",
            "url": "https://example.com/guide",
            "snippet": "Canvas and nylon comparison",
        }
    ]


@pytest.mark.asyncio
async def test_openai_llm_provider_parses_action_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer unit-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"action":"item_search","arguments":{"query":"x"}}'}}
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAILLMProvider(api_key="unit-key", model="unit-model", client=client)

    result = await provider.plan_next_action([{"role": "user", "content": "x"}])

    assert result.provider_mode == "real"
    assert result.data == {"action": "item_search", "arguments": {"query": "x"}}


def test_registry_selects_real_http_providers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("OMNIMATCH_PRODUCT_API_KEY", "product-key")
    monkeypatch.setenv("OMNIMATCH_WEB_SEARCH_API_KEY", "search-key")
    settings = OmniMatchSettings(
        profile="dev",
        llm_provider="openai",
        llm_model="unit-model",
        product_provider="http_product",
        web_search_provider="http_web_search",
        shipping_provider="placeholder",
        memory_provider="memory",
        eval_provider="heuristic",
        product_api_url="https://product.example/search",
        web_search_api_url="https://search.example/query",
    )

    registry = ProviderRegistry.from_settings(settings)

    assert isinstance(registry.llm, OpenAILLMProvider)
    assert registry.llm.base_url == "https://llm.example/v1"
    assert isinstance(registry.product, HttpProductSearchProvider)
    assert isinstance(registry.web_search, HttpWebSearchProvider)


def test_registry_selects_serper_web_search_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OMNIMATCH_PRODUCT_API_KEY", "product-key")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    settings = OmniMatchSettings(
        profile="dev",
        llm_provider="openai",
        llm_model="unit-model",
        product_provider="http_product",
        web_search_provider="serper",
        shipping_provider="placeholder",
        memory_provider="memory",
        eval_provider="heuristic",
        product_api_url="https://product.example/search",
    )

    registry = ProviderRegistry.from_settings(settings)

    assert isinstance(registry.web_search, SerperWebSearchProvider)


def test_registry_selects_serpapi_product_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-key")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    settings = OmniMatchSettings(
        profile="dev",
        llm_provider="openai",
        llm_model="unit-model",
        product_provider="serpapi",
        web_search_provider="serper",
        shipping_provider="placeholder",
        memory_provider="memory",
        eval_provider="heuristic",
    )

    registry = ProviderRegistry.from_settings(settings)

    assert isinstance(registry.product, SerpApiProductProvider)


@pytest.mark.asyncio
async def test_rate_table_shipping_provider_estimates_local_rates():
    provider = RateTableShippingProvider()

    result = await provider.estimate(
        {"price": 198, "platform": "Amazon"},
        destination="CN",
    )

    assert result.provider == "rate_table"
    assert result.provider_mode == "real"
    assert result.data == {"shipping": 25, "tax": 0, "destination": "CN"}


def test_registry_selects_rate_table_shipping_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-key")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    settings = OmniMatchSettings(
        profile="dev",
        llm_provider="openai",
        llm_model="unit-model",
        product_provider="serpapi",
        web_search_provider="serper",
        shipping_provider="rate_table",
        memory_provider="memory",
        eval_provider="heuristic",
    )

    registry = ProviderRegistry.from_settings(settings)

    assert isinstance(registry.shipping, RateTableShippingProvider)
