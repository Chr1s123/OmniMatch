from __future__ import annotations

from typing import Any

from app.schemas import ProductCandidate, ScoredProduct, ShoppingIntent
from app.tools.category_insight import get_category_insight
from app.tools.context import ToolContext
from app.tools.item_picker import pick_items
from app.tools.item_search import search_items
from app.tools.planner import plan_query
from app.tools.price_compare import compare_prices
from app.tools.shipping_calc import calculate_shipping


class ToolRegistry:
    def __init__(
        self,
        ctx: ToolContext,
        allowed_tools: frozenset[str] | None = None,
    ) -> None:
        self.ctx = ctx
        self.allowed_tools = allowed_tools
        self.intent: ShoppingIntent | None = None
        self.insight: dict[str, Any] | None = None
        self.candidates: list[ProductCandidate] = []
        self.scored: list[ScoredProduct] = []

    async def run(self, action: str, arguments: dict[str, Any]) -> object:
        if self.allowed_tools is not None and action not in self.allowed_tools:
            raise PermissionError(f"tool action is not allowed in this agent scope: {action}")
        if action == "plan":
            self.intent = await plan_query(arguments["query"], self.ctx)
            return self.intent
        if action == "category_insight":
            self._require_intent()
            self.insight = await get_category_insight(self.intent, self.ctx)
            return self.insight
        if action == "item_search":
            self._require_intent()
            self.candidates = await search_items(self.intent, self.insight or {}, self.ctx)
            return self.candidates
        if action == "shipping":
            self.candidates = await calculate_shipping(self.candidates, self.ctx)
            return self.candidates
        if action == "rank":
            self._require_intent()
            self.scored = await compare_prices(self.candidates, self.intent, self.ctx)
            return self.scored
        if action == "pick":
            self._require_intent()
            return await pick_items(self.scored, self.intent, self.ctx)
        raise ValueError(f"unknown tool action: {action}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "has_intent": self.intent is not None,
            "has_insight": self.insight is not None,
            "candidate_count": len(self.candidates),
            "scored_count": len(self.scored),
            "top_score": self.scored[0].score.total if self.scored else None,
        }

    def _require_intent(self) -> None:
        if self.intent is None:
            raise ValueError("plan action must run before this tool")
