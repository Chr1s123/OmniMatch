import re


async def plan_query(query: str) -> dict:
    budget_match = re.search(r"预算\s*(\d+)|(\d+)\s*块|(\d+)\s*元", query)
    budget = 300
    if budget_match:
        budget = int(next(group for group in budget_match.groups() if group))

    preferences: list[str] = []
    if "不要塑料" in query or "非塑料" in query:
        preferences.append("不要塑料")
    if "小众" in query:
        preferences.append("小众")
    if "抗造" in query or "耐用" in query:
        preferences.append("耐用")

    category = "旅行三件套" if "旅行" in query else "通用商品"
    return {
        "original_query": query,
        "category": category,
        "budget": budget,
        "preferences": preferences,
    }
