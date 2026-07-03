from __future__ import annotations

import asyncio
from uuid import uuid4

from app.api.monitor import EventCollector
from app.schemas import Product
from app.tools.item_search import search_items
from app.tools.shipping_calc import calculate_shipping


PLATFORMS = ["Amazon", "eBay", "AliExpress", "Shopee"]


async def dispatch_platform_search(
    intent: dict,
    insight: dict,
    monitor: EventCollector,
    platforms: list[str] | None = None,
) -> list[Product]:
    selected_platforms = platforms or PLATFORMS
    results = await asyncio.gather(
        *[_search_platform(platform, intent, insight, monitor) for platform in selected_platforms]
    )
    return [product for platform_products in results for product in platform_products]


async def _search_platform(
    platform: str,
    intent: dict,
    insight: dict,
    monitor: EventCollector,
) -> list[Product]:
    subagent_id = f"sub-{uuid4().hex[:8]}"
    await monitor.emit(
        "subagent_started",
        f"{platform} 子 Agent 开始检索商品...",
        tool="dispatch_tool",
        payload={"subagent_id": subagent_id, "platform": platform},
    )
    products = await search_items(platform, intent, insight)
    products = await calculate_shipping(products)
    await monitor.emit(
        "subagent_finished",
        f"{platform} 子 Agent 返回 {len(products)} 件商品。",
        tool="dispatch_tool",
        payload={"subagent_id": subagent_id, "platform": platform, "count": len(products)},
    )
    return products
