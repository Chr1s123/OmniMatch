import httpx
import pytest

from app.config import OmniMatchSettings
from app.providers.http_product import HttpProductSearchProvider
from app.providers.http_web_search import HttpWebSearchProvider
from app.providers.openai_llm import OpenAILLMProvider
from app.providers.registry import ProviderRegistry


@pytest.mark.asyncio
async def test_http_product_provider_normalizes_items():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "unit-key"
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
