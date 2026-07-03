async def get_category_insight(intent: dict) -> dict:
    category = intent.get("category", "通用商品")
    return {
        "category": category,
        "popular_attributes": ["轻量", "耐用", "易收纳"],
        "avoid_attributes": ["塑料"] if "不要塑料" in intent.get("preferences", []) else [],
        "price_band": "100-300 CNY",
    }
