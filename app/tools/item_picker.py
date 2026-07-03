from app.schemas import Product


async def pick_items(products: list[Product], intent: dict) -> list[Product]:
    budget = intent.get("budget", 300)
    within_budget = [product for product in products if product.total_price <= budget]
    candidates = within_budget or products
    return candidates[:3]
