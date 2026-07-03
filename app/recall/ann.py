def search_ann(vector: list[float], limit: int = 5) -> list[dict]:
    return [{"id": f"mock-{index}", "score": 1.0 / (index + 1)} for index in range(limit)]
