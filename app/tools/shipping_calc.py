from app.schemas import ProductCandidate
from app.tools.context import ToolContext


async def calculate_shipping(
    candidates: list[ProductCandidate],
    ctx: ToolContext,
) -> list[ProductCandidate]:
    priced: list[ProductCandidate] = []
    for candidate in candidates:
        result = await ctx.providers.shipping.estimate(
            candidate.model_dump(),
            destination=None,
        )
        ctx.observations.append(
            {
                "tool": "ShippingCalc",
                "provider": result.provider,
                "provider_mode": result.provider_mode,
                "latency_ms": result.latency_ms,
                "warnings": result.warnings,
            }
        )
        priced.append(
            candidate.model_copy(
                update={
                    "shipping": result.data.get("shipping", 0),
                    "tax": result.data.get("tax", 0),
                }
            )
        )
    return priced
