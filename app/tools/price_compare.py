from app.schemas import Product


async def compare_prices(products: list[Product]) -> list[Product]:
    return sorted(products, key=lambda product: (product.total_price, -product.rating))
