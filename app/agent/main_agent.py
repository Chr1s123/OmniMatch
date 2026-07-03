from __future__ import annotations

import json
from pathlib import Path

from app.agent.tool_registry import ToolRegistry
from app.api.context import set_task_context
from app.api.monitor import EventCollector
from app.config import OmniMatchSettings
from app.providers.registry import ProviderRegistry
from app.schemas import ShoppingSummary
from app.tools.context import ToolContext
from app.tools.shopping_summary import build_summary


class CompetitionAgentLoop:
    def __init__(
        self,
        thread_id: str,
        session_dir: str | Path,
        settings: OmniMatchSettings,
        providers: ProviderRegistry,
        monitor: EventCollector,
    ) -> None:
        self.thread_id = thread_id
        self.session_dir = Path(session_dir)
        self.settings = settings
        self.providers = providers
        self.monitor = monitor

    async def run(self, query: str) -> ShoppingSummary:
        set_task_context(self.thread_id, self.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(settings=self.settings, providers=self.providers)
        tools = ToolRegistry(ctx)
        trace: list[dict] = []

        await self.monitor.emit(
            "task_started",
            "Competition Agent started.",
            payload={
                "profile": self.settings.profile,
                "provider_modes": self.settings.provider_modes(),
            },
        )

        picked = None
        for action, arguments in [
            ("plan", {"query": query}),
            ("category_insight", {}),
            ("item_search", {}),
            ("shipping", {}),
            ("rank", {}),
            ("pick", {}),
        ]:
            await self.monitor.emit("tool_start", f"{action} started", tool=action)
            observation_start = len(ctx.observations)
            result = await tools.run(action, arguments)
            new_observations = ctx.observations[observation_start:]
            trace.append(
                {
                    "action": action,
                    "observation_count": len(ctx.observations),
                    "observations": new_observations,
                }
            )
            for observation in new_observations:
                if observation.get("provider"):
                    await self.monitor.emit(
                        "provider_start",
                        f"{observation['provider']} used by {observation.get('tool', action)}.",
                        tool=action,
                        payload={
                            "provider": observation.get("provider"),
                            "provider_mode": observation.get("provider_mode"),
                        },
                    )
                    await self.monitor.emit(
                        "provider_end",
                        f"{observation['provider']} completed.",
                        tool=action,
                        payload=observation,
                    )
            await self.monitor.emit("tool_end", f"{action} finished", tool=action)
            if action == "rank":
                await self.monitor.emit(
                    "ranking_decision",
                    "Candidates scored.",
                    payload={"candidate_count": len(result)},
                )
            if action == "pick":
                picked = result

        summary = await build_summary(query, picked or [], ctx)
        try:
            self._write_json("summary.json", summary.model_dump())
            self._write_json("candidates.json", [item.model_dump() for item in tools.scored])
            self._write_jsonl("trace.jsonl", trace)
        except OSError as exc:
            summary.warnings.append(f"output persistence failed: {exc}")
            await self.monitor.emit(
                "task_warning",
                f"Output persistence failed: {exc}",
                payload={"warning": str(exc)},
            )
        await self.monitor.emit(
            "task_result",
            "Shopping summary generated.",
            payload={"summary": summary.model_dump()},
        )
        return summary

    def _write_json(self, filename: str, payload: object) -> None:
        (self.session_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_jsonl(self, filename: str, rows: list[dict]) -> None:
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        (self.session_dir / filename).write_text(text + "\n", encoding="utf-8")


MockAgentLoop = CompetitionAgentLoop
