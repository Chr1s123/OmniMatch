from app.schemas import ShoppingIntent
from app.tools.context import ToolContext


async def get_category_insight(intent: ShoppingIntent, ctx: ToolContext) -> dict:
    result = await ctx.providers.web_search.search(f"{intent.category} buying guide")
    ctx.observations.append(
        {
            "tool": "CategoryInsight",
            "provider": result.provider,
            "provider_mode": result.provider_mode,
            "latency_ms": result.latency_ms,
            "warnings": result.warnings,
        }
    )
    return {
        "category": intent.category,
        "popular_attributes": ["轻量", "耐用", "易收纳"],
        "avoid_attributes": intent.negative_constraints,
        "price_band": "100-300 CNY",
        "platforms": ["Amazon", "eBay", "AliExpress", "Shopee"],
        "evidence": result.data,
    }
