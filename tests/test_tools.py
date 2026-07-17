import pytest

from app.config import OmniMatchSettings
from app.providers.base import ProviderResult
from app.providers.registry import ProviderRegistry
from app.agent.tool_registry import ToolRegistry
from app.schemas import CandidateScore, ProductCandidate, ScoredProduct, ShoppingIntent
from app.tools.category_insight import get_category_insight
from app.tools.context import ToolContext
from app.tools.item_picker import pick_items
from app.tools.item_search import search_items
from app.tools.planner import plan_query
from app.tools.price_compare import compare_prices
from app.tools.shipping_calc import calculate_shipping
from app.tools.shopping_summary import build_summary


def submission_settings() -> OmniMatchSettings:
    return OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="placeholder",
    )


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


@pytest.mark.asyncio
async def test_tool_registry_snapshot_reports_progress():
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
    registry = ToolRegistry(ctx)

    initial = registry.snapshot()
    assert initial == {
        "has_intent": False,
        "has_insight": False,
        "candidate_count": 0,
        "scored_count": 0,
        "top_score": None,
    }

    await registry.run("plan", {"query": "旅行三件套，预算300，不要塑料"})
    await registry.run("category_insight", {})
    await registry.run("item_search", {})
    await registry.run("shipping", {})
    await registry.run("rank", {})

    after_rank = registry.snapshot()
    assert after_rank["has_intent"] is True
    assert after_rank["has_insight"] is True
    assert after_rank["candidate_count"] == 4
    assert after_rank["scored_count"] == 4
    assert after_rank["top_score"] is not None


@pytest.mark.asyncio
async def test_tool_registry_rejects_tools_outside_child_allowlist():
    settings = submission_settings()
    ctx = ToolContext(settings=settings, providers=ProviderRegistry.from_settings(settings))
    registry = ToolRegistry(ctx, allowed_tools=frozenset({"plan"}))

    await registry.run("plan", {"query": "旅行三件套"})

    with pytest.raises(PermissionError, match="item_search"):
        await registry.run("item_search", {})


@pytest.mark.asyncio
async def test_pick_items_skips_missing_urls_and_backfills_from_later_candidates():
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
    intent = ShoppingIntent(original_query="旅行三件套", category="旅行三件套")
    scored = [
        scored_candidate("no-url-1", "", 90),
        scored_candidate("with-url-1", "https://example.com/1", 80),
        scored_candidate("no-url-2", "  ", 70),
        scored_candidate("with-url-2", "https://example.com/2", 60),
        scored_candidate("with-url-3", "https://example.com/3", 50),
    ]

    picked = await pick_items(scored, intent, ctx)

    assert [item.candidate.id for item in picked] == ["with-url-1", "with-url-2", "with-url-3"]
    assert all(item.candidate.url.strip() for item in picked)
    assert ctx.observations[-1] == {
        "tool": "ItemPicker",
        "picked_count": 3,
        "skipped_missing_url_count": 2,
    }


@pytest.mark.asyncio
async def test_summary_reports_when_all_candidates_are_missing_urls():
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
    intent = ShoppingIntent(original_query="旅行三件套", category="旅行三件套")
    scored = [
        scored_candidate("no-url-1", "", 90),
        scored_candidate("no-url-2", "  ", 80),
    ]

    picked = await pick_items(scored, intent, ctx)
    summary = await build_summary("旅行三件套", picked, ctx)

    assert summary.products == []
    assert "没有找到带可跳转链接" in summary.message
    assert "skipped 2 candidates without product URLs" in summary.warnings


def scored_candidate(candidate_id: str, url: str, total: float) -> ScoredProduct:
    candidate = ProductCandidate(
        id=candidate_id,
        platform="Unit",
        title=f"Candidate {candidate_id}",
        price=100,
        rating=4.5,
        url=url,
    )
    score = CandidateScore(
        total=total,
        constraint_score=total,
        evidence_score=0,
        price_score=0,
        preference_score=0,
        risk_penalty=0,
        total_landed_cost=candidate.total_landed_cost,
    )
    return ScoredProduct(candidate=candidate, score=score)
