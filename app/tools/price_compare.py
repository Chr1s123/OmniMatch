from app.ranking.scorer import score_candidates
from app.schemas import ProductCandidate, ScoredProduct, ShoppingIntent
from app.tools.context import ToolContext


async def compare_prices(
    candidates: list[ProductCandidate],
    intent: ShoppingIntent,
    ctx: ToolContext,
) -> list[ScoredProduct]:
    scored = score_candidates(intent, candidates)
    ctx.observations.append({"tool": "PriceCompare", "candidate_count": len(scored)})
    return scored
