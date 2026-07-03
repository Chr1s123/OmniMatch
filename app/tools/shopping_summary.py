from app.schemas import Product, ShoppingSummary


async def build_summary(query: str, products: list[Product]) -> ShoppingSummary:
    count = len(products)
    return ShoppingSummary(
        message=f"基于“{query}”，为你推荐 {count} 件 mock 商品，已按含运费总价和偏好排序。",
        products=products,
    )
