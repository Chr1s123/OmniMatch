import pytest

from app.tools.category_insight import get_category_insight
from app.tools.item_picker import pick_items
from app.tools.item_search import search_items
from app.tools.planner import plan_query
from app.tools.price_compare import compare_prices
from app.tools.shipping_calc import calculate_shipping
from app.tools.shopping_summary import build_summary


@pytest.mark.asyncio
async def test_mock_tool_chain_returns_ranked_summary():
    intent = await plan_query("我想买旅行三件套，预算300，不要塑料")
    assert intent["budget"] == 300
    assert "不要塑料" in intent["preferences"]

    insight = await get_category_insight(intent)
    assert "旅行" in insight["category"]

    products = await search_items("Amazon", intent, insight)
    assert products[0].platform == "Amazon"

    shipped = await calculate_shipping(products)
    assert all(product.shipping >= 0 for product in shipped)

    compared = await compare_prices(shipped)
    assert compared == sorted(compared, key=lambda product: product.total_price)

    picked = await pick_items(compared, intent)
    assert len(picked) <= 3

    summary = await build_summary("原始需求", picked)
    assert summary.products == picked
    assert "原始需求" in summary.message
