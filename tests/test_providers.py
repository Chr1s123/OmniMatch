import pytest

from app.config import OmniMatchSettings
from app.providers.base import ProviderResult
from app.providers.placeholder import PlaceholderLLMProvider
from app.providers.registry import ProviderRegistry


@pytest.mark.asyncio
async def test_submission_registry_uses_placeholder_providers():
    settings = OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="placeholder",
    )

    registry = ProviderRegistry.from_settings(settings)
    result = await registry.product.search("旅行三件套", platforms=["Amazon"])

    assert isinstance(result, ProviderResult)
    assert result.provider_mode == "placeholder"
    assert result.latency_ms >= 0
    assert result.data
    assert "api_key" not in result.response_summary.lower()


def test_provider_result_redacts_secret_like_values():
    result = ProviderResult(
        provider="unit",
        provider_mode="real",
        latency_ms=1,
        data={"ok": True},
        warnings=[],
        response_summary="Authorization: Bearer secret-token api_key=abc",
    )

    assert "secret-token" not in result.redacted_summary()
    assert "api_key=abc" not in result.redacted_summary()


@pytest.mark.asyncio
async def test_placeholder_llm_proposes_deterministic_action_sequence():
    provider = PlaceholderLLMProvider()
    actions: list[str] = []

    for _ in range(7):
        result = await provider.plan_next_action([{"role": "user", "content": "next"}])
        actions.append(result.data["action"])

    assert actions == [
        "plan",
        "category_insight",
        "item_search",
        "shipping",
        "rank",
        "pick",
        "finish",
    ]
