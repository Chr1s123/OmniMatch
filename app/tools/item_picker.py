from app.schemas import ScoredProduct, ShoppingIntent
from app.tools.context import ToolContext


async def pick_items(
    scored: list[ScoredProduct],
    intent: ShoppingIntent,
    ctx: ToolContext,
) -> list[ScoredProduct]:
    picked = scored[:3]
    ctx.observations.append({"tool": "ItemPicker", "picked_count": len(picked)})
    return picked
