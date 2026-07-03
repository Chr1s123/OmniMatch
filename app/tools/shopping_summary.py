from app.schemas import Product, ScoredProduct, ShoppingSummary
from app.tools.context import ToolContext


async def build_summary(
    query: str,
    picked: list[ScoredProduct],
    ctx: ToolContext,
) -> ShoppingSummary:
    products = [
        Product(
            id=item.candidate.id,
            platform=item.candidate.platform,
            title=item.candidate.title,
            price=item.candidate.price,
            currency=item.candidate.currency,
            shipping=item.candidate.shipping,
            tax=item.candidate.tax,
            rating=item.candidate.rating,
            reason=_reason_for(item),
            url=item.candidate.url,
        )
        for item in picked
    ]
    count = len(products)
    provider_modes = ", ".join(
        sorted({str(obs.get("provider_mode")) for obs in ctx.observations if obs.get("provider_mode")})
    )
    return ShoppingSummary(
        message=f"基于“{query}”，为你推荐 {count} 件商品，已按约束、证据和含运费总价排序。",
        products=products,
        warnings=[f"evidence used provider modes: {provider_modes}"] if provider_modes else [],
    )


def _reason_for(item: ScoredProduct) -> str:
    reasons = item.score.rejection_reasons
    if reasons:
        return f"评分 {item.score.total}，注意：{'; '.join(reasons)}"
    return f"评分 {item.score.total}，证据充分且约束匹配。"
