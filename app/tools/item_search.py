from app.schemas import Product


PLATFORM_PRICE_OFFSET = {
    "Amazon": 0,
    "eBay": -12,
    "AliExpress": -35,
    "Shopee": -20,
}


async def search_items(platform: str, intent: dict, insight: dict) -> list[Product]:
    offset = PLATFORM_PRICE_OFFSET.get(platform, 0)
    category = insight.get("category", intent.get("category", "旅行用品"))
    slug = platform.lower().replace(" ", "-")
    return [
        Product(
            id=f"{slug}-canvas-set",
            platform=platform,
            title=f"{platform} 帆布{category}",
            price=198 + offset,
            currency="CNY",
            rating=4.6,
            reason="帆布材质，避开塑料，适合耐用优先的需求",
            url=f"https://example.com/{slug}/canvas-set",
        ),
        Product(
            id=f"{slug}-organizer-set",
            platform=platform,
            title=f"{platform} 轻量收纳{category}",
            price=168 + offset,
            currency="CNY",
            rating=4.3,
            reason="价格低，收纳友好，适合作为预算优先备选",
            url=f"https://example.com/{slug}/organizer-set",
        ),
    ]
