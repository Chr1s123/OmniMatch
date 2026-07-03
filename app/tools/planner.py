import re

from app.schemas import ShoppingIntent
from app.tools.context import ToolContext


async def plan_query(query: str, ctx: ToolContext) -> ShoppingIntent:
    budget_match = re.search(r"预算\s*(\d+)|(\d+)\s*块|(\d+)\s*元", query)
    budget: float | None = 300
    if budget_match:
        budget = int(next(group for group in budget_match.groups() if group))

    preferences: list[str] = []
    negative_constraints: list[str] = []
    if "不要塑料" in query or "非塑料" in query:
        negative_constraints.append("塑料")
    if "小众" in query:
        preferences.append("小众")
    if "抗造" in query or "耐用" in query:
        preferences.append("耐用")

    category = "旅行三件套" if "旅行" in query else "通用商品"
    intent = ShoppingIntent(
        original_query=query,
        category=category,
        budget=budget,
        preferences=preferences,
        negative_constraints=negative_constraints,
    )
    ctx.observations.append({"tool": "Planner", "category": category, "budget": budget})
    return intent
