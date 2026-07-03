from app.schemas import ProductCandidate, ShoppingIntent
from app.tools.context import ToolContext


DEFAULT_PLATFORMS = ["Amazon", "eBay", "AliExpress", "Shopee"]


async def search_items(
    intent: ShoppingIntent,
    insight: dict,
    ctx: ToolContext,
) -> list[ProductCandidate]:
    query = f"{intent.category} {' '.join(intent.preferences)}".strip()
    platforms = insight.get("platforms", DEFAULT_PLATFORMS)
    result = await ctx.providers.product.search(query, platforms=platforms)
    ctx.observations.append(
        {
            "tool": "ItemSearch",
            "provider": result.provider,
            "provider_mode": result.provider_mode,
            "latency_ms": result.latency_ms,
            "warnings": result.warnings,
        }
    )
    return [ProductCandidate(**item) for item in result.data]
