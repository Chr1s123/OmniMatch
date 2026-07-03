from __future__ import annotations

import json
from pathlib import Path

from app.agent.dispatch_tool import dispatch_platform_search
from app.api.context import set_task_context
from app.api.monitor import EventCollector
from app.schemas import Product, ShoppingSummary
from app.tools.category_insight import get_category_insight
from app.tools.item_picker import pick_items
from app.tools.planner import plan_query
from app.tools.price_compare import compare_prices
from app.tools.shopping_summary import build_summary


class MockAgentLoop:
    def __init__(self, thread_id: str, session_dir: str | Path, monitor: EventCollector) -> None:
        self.thread_id = thread_id
        self.session_dir = Path(session_dir)
        self.monitor = monitor

    async def run(self, query: str) -> ShoppingSummary:
        set_task_context(self.thread_id, self.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        await self.monitor.emit("task_started", "OmniMatch mock AgentLoop 已启动。")
        await self.monitor.emit("thought", "Think: 正在理解购物意图。")

        intent = await self._run_tool("Planner", "Planner 正在拆解需求...", plan_query, query)
        insight = await self._run_tool(
            "CategoryInsight",
            "CategoryInsight 正在生成品类洞察...",
            get_category_insight,
            intent,
        )

        await self.monitor.emit(
            "tool_start",
            "ItemSearch 正在跨 4 个平台并行检索...",
            tool="ItemSearch",
            payload={"platforms": ["Amazon", "eBay", "AliExpress", "Shopee"]},
        )
        products = await dispatch_platform_search(intent, insight, self.monitor)
        await self.monitor.emit(
            "tool_end",
            f"ItemSearch 合流完成，共获得 {len(products)} 件商品。",
            tool="ItemSearch",
            payload={"count": len(products)},
        )

        compared = await self._run_tool(
            "PriceCompare",
            "PriceCompare 正在计算含运费总价...",
            compare_prices,
            products,
        )
        picked = await self._run_tool(
            "ItemPicker",
            "ItemPicker 正在按预算和偏好精挑...",
            pick_items,
            compared,
            intent,
        )
        summary = await self._run_tool(
            "ShoppingSummary",
            "ShoppingSummary 正在生成最终清单...",
            build_summary,
            query,
            picked,
        )

        self._write_summary(summary)
        await self.monitor.emit(
            "task_result",
            "购物清单已生成。",
            payload={"summary": summary.model_dump()},
        )
        return summary

    async def _run_tool(self, tool: str, message: str, fn, *args):
        await self.monitor.emit("tool_start", message, tool=tool)
        result = await fn(*args)
        payload = self._payload_for_result(result)
        await self.monitor.emit("tool_end", f"{tool} 执行完成。", tool=tool, payload=payload)
        return result

    def _payload_for_result(self, result) -> dict:
        if isinstance(result, ShoppingSummary):
            return {"product_count": len(result.products)}
        if isinstance(result, list) and all(isinstance(item, Product) for item in result):
            return {"count": len(result)}
        if isinstance(result, dict):
            return result
        return {}

    def _write_summary(self, summary: ShoppingSummary) -> None:
        path = self.session_dir / "summary.json"
        path.write_text(
            json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
