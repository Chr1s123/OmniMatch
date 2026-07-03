import pytest

from app.config import OmniMatchSettings
from app.providers.base import ProviderResult
from app.providers.registry import ProviderRegistry
from app.tools.category_insight import get_category_insight
from app.tools.context import ToolContext
from app.tools.item_picker import pick_items
from app.tools.item_search import search_items
from app.tools.planner import plan_query
from app.tools.price_compare import compare_prices
from app.tools.shipping_calc import calculate_shipping
from app.tools.shopping_summary import build_summary


class RecordingLLMProvider:
    def __init__(self) -> None:
        self.messages: list[dict] | None = None

    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        self.messages = messages
        return ProviderResult(
            provider="unit_llm",
            provider_mode="real",
            latency_ms=7,
            data={
                "action": "plan_query",
                "arguments": {
                    "category": "登山包",
                    "budget": 500,
                    "preferences": ["防水"],
                    "negative_constraints": ["皮革"],
                    "destination": "Shanghai",
                },
            },
        )


@pytest.mark.asyncio
async def test_tool_chain_uses_provider_backed_candidates():
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
    ctx = ToolContext(settings=settings, providers=ProviderRegistry.from_settings(settings))

    intent = await plan_query("我想买旅行三件套，预算300，不要塑料", ctx)
    insight = await get_category_insight(intent, ctx)
    candidates = await search_items(intent, insight, ctx)
    shipped = await calculate_shipping(candidates, ctx)
    compared = await compare_prices(shipped, intent, ctx)
    picked = await pick_items(compared, intent, ctx)
    summary = await build_summary("原始需求", picked, ctx)

    assert intent.negative_constraints == ["塑料"]
    assert candidates[0].evidence
    assert picked[0].score.total >= picked[-1].score.total
    assert len(picked) <= 3
    assert summary.products


@pytest.mark.asyncio
async def test_plan_query_uses_llm_provider_for_intent():
    settings = OmniMatchSettings(
        profile="dev",
        llm_provider="openai",
        llm_model="unit-model",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="memory",
        eval_provider="heuristic",
    )
    base_registry = ProviderRegistry.from_settings(settings)
    llm = RecordingLLMProvider()
    providers = ProviderRegistry(
        llm=llm,
        product=base_registry.product,
        web_search=base_registry.web_search,
        shipping=base_registry.shipping,
    )
    ctx = ToolContext(settings=settings, providers=providers)

    intent = await plan_query("我想买一个防水登山包，预算500，不要皮革", ctx)

    assert llm.messages is not None
    assert intent.category == "登山包"
    assert intent.budget == 500
    assert intent.preferences == ["防水"]
    assert intent.negative_constraints == ["皮革"]
    assert intent.destination == "Shanghai"
    assert ctx.observations[-1] == {
        "tool": "Planner",
        "provider": "unit_llm",
        "provider_mode": "real",
        "latency_ms": 7,
        "warnings": [],
    }
