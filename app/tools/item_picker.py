from app.schemas import ScoredProduct, ShoppingIntent
from app.tools.context import ToolContext


async def pick_items(
    scored: list[ScoredProduct],
    intent: ShoppingIntent,
    ctx: ToolContext,
) -> list[ScoredProduct]:
    linkable_items = [item for item in scored if item.candidate.url.strip()]
    picked = linkable_items[:3]
    skipped_missing_url_count = len(scored) - len(linkable_items)
    ctx.observations.append(
        {
            "tool": "ItemPicker",
            "picked_count": len(picked),
            "skipped_missing_url_count": skipped_missing_url_count,
        }
    )
    return picked
