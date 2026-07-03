from app.schemas import AgentEvent, Product, ShoppingQuery, ShoppingSummary, TaskState


def test_shopping_query_strips_query_text():
    query = ShoppingQuery(query="  旅行三件套，不要塑料  ")
    assert query.query == "旅行三件套，不要塑料"


def test_product_total_price_includes_shipping_and_tax():
    product = Product(
        id="p1",
        platform="Amazon",
        title="旅行收纳三件套",
        price=199.0,
        currency="CNY",
        shipping=20.0,
        tax=5.0,
        rating=4.6,
        reason="便宜耐用",
        url="https://example.com/p1",
    )
    assert product.total_price == 224.0


def test_task_state_defaults_to_running():
    state = TaskState(thread_id="thread_abc")
    assert state.status == "running"
    assert state.events == []


def test_summary_contains_products_and_message():
    product = Product(
        id="p1",
        platform="eBay",
        title="帆布旅行套装",
        price=180,
        currency="CNY",
        shipping=30,
        tax=0,
        rating=4.4,
        reason="材质非塑料",
        url="https://example.com/p1",
    )
    summary = ShoppingSummary(message="推荐 1 件商品", products=[product])
    assert summary.products[0].total_price == 210
    assert "推荐" in summary.message


def test_agent_event_has_display_fields():
    event = AgentEvent(
        type="tool_start",
        thread_id="thread_abc",
        run_id="run_abc",
        tool="Planner",
        message="Planner 正在拆解需求...",
    )
    assert event.type == "tool_start"
    assert event.payload == {}
