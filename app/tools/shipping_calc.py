from app.schemas import Product


PLATFORM_SHIPPING = {
    "Amazon": 28,
    "eBay": 35,
    "AliExpress": 18,
    "Shopee": 22,
}


async def calculate_shipping(products: list[Product]) -> list[Product]:
    priced: list[Product] = []
    for product in products:
        shipping = PLATFORM_SHIPPING.get(product.platform, 25)
        tax = round(product.price * 0.03, 2) if product.price > 180 else 0
        priced.append(product.model_copy(update={"shipping": shipping, "tax": tax}))
    return priced
