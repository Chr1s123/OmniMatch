from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agent.actions import AgentAction, AgentStep
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
        max_steps: int = 8,
    ) -> None:
        self.thread_id = thread_id
        self.session_dir = Path(session_dir)
        self.settings = settings
        self.providers = providers
        self.monitor = monitor
        self.max_steps = max_steps

    async def run(self, query: str) -> ShoppingSummary:
        set_task_context(self.thread_id, self.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(settings=self.settings, providers=self.providers)
        tools = ToolRegistry(ctx)
        trace: list[dict[str, Any]] = []
        steps: list[AgentStep] = []
        picked = None
        terminal_note = ""

        await self.monitor.emit(
            "task_started",
            "Competition Agent started.",
            payload={
                "profile": self.settings.profile,
                "provider_modes": self.settings.provider_modes(),
            },
        )

        for step_index in range(self.max_steps):
            action, planner_observation = await self._plan_next_action(
                query=query,
                step_index=step_index,
                tools=tools,
                ctx=ctx,
                steps=steps,
            )
            await self._emit_provider_observation(action.name, planner_observation)
            await self.monitor.emit(
                "thought",
                action.thought or action.message or f"Selected action {action.name}.",
                payload={"action": action.name, "arguments": action.arguments},
            )

            if action.name == "finish":
                terminal_note = action.message or terminal_note
                steps.append(
                    AgentStep(
                        action=action,
                        observation_count=len(ctx.observations),
                        observations=[planner_observation],
                    )
                )
                trace.append(self._trace_row(action, ctx.observations[-1:], len(ctx.observations)))
                break

            if action.name == "clarify":
                terminal_note = action.message or "需要更多信息才能给出可靠推荐。"
                steps.append(
                    AgentStep(
                        action=action,
                        observation_count=len(ctx.observations),
                        observations=[planner_observation],
                    )
                )
                trace.append(self._trace_row(action, [planner_observation], len(ctx.observations)))
                break

            if action.name == "fail":
                terminal_note = action.message or "代理无法继续执行。"
                steps.append(
                    AgentStep(
                        action=action,
                        observation_count=len(ctx.observations),
                        observations=[planner_observation],
                    )
                )
                trace.append(self._trace_row(action, [planner_observation], len(ctx.observations)))
                await self.monitor.emit("task_error", terminal_note)
                break

            await self.monitor.emit("tool_start", f"{action.name} started", tool=action.name)
            observation_start = len(ctx.observations)
            result = await tools.run(action.name, self._tool_arguments(action, query))
            tool_observations = ctx.observations[observation_start:]
            for observation in tool_observations:
                await self._emit_provider_observation(action.name, observation)
            await self.monitor.emit("tool_end", f"{action.name} finished", tool=action.name)

            if action.name == "rank":
                await self.monitor.emit(
                    "ranking_decision",
                    "Candidates scored.",
                    payload={"candidate_count": len(result)},
                )
            if action.name == "pick":
                picked = result

            step_observations = [planner_observation, *tool_observations]
            steps.append(
                AgentStep(
                    action=action,
                    observation_count=len(ctx.observations),
                    observations=step_observations,
                )
            )
            trace.append(self._trace_row(action, step_observations, len(ctx.observations)))
        else:
            terminal_note = f"Reached max_steps={self.max_steps} before a finish action."
            await self.monitor.emit("task_error", terminal_note)

        summary = await build_summary(query, picked or [], ctx, status_note=terminal_note)
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

    async def _plan_next_action(
        self,
        query: str,
        step_index: int,
        tools: ToolRegistry,
        ctx: ToolContext,
        steps: list[AgentStep],
    ) -> tuple[AgentAction, dict[str, Any]]:
        result = await self.providers.llm.plan_next_action(
            [
                {
                    "role": "system",
                    "content": (
                        "Choose the next shopping-agent action as JSON. Allowed tool actions: "
                        "plan, category_insight, item_search, shipping, rank, pick. "
                        "Allowed terminal actions: finish, clarify, fail. "
                        "Return action, arguments, thought, and optional message."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "step_index": step_index,
                            "tool_state": tools.snapshot(),
                            "recent_observations": ctx.observations[-5:],
                            "completed_actions": [step.action.name for step in steps],
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        planner_observation = {
            "tool": "AgentPlanner",
            "provider": result.provider,
            "provider_mode": result.provider_mode,
            "latency_ms": result.latency_ms,
            "warnings": result.warnings,
        }
        ctx.observations.append(planner_observation)
        return AgentAction.from_provider_data(result.data), planner_observation

    def _tool_arguments(self, action: AgentAction, query: str) -> dict[str, Any]:
        if action.name == "plan":
            return {"query": action.arguments.get("query") or query}
        return action.arguments

    async def _emit_provider_observation(
        self,
        action_name: str,
        observation: dict[str, Any],
    ) -> None:
        if not observation.get("provider"):
            return
        await self.monitor.emit(
            "provider_start",
            f"{observation['provider']} used by {observation.get('tool', action_name)}.",
            tool=action_name,
            payload={
                "provider": observation.get("provider"),
                "provider_mode": observation.get("provider_mode"),
            },
        )
        await self.monitor.emit(
            "provider_end",
            f"{observation['provider']} completed.",
            tool=action_name,
            payload=observation,
        )

    def _trace_row(
        self,
        action: AgentAction,
        observations: list[dict[str, Any]],
        observation_count: int,
    ) -> dict[str, Any]:
        return {
            "action": action.name,
            "thought": action.thought,
            "message": action.message,
            "observation_count": observation_count,
            "observations": observations,
        }

    def _write_json(self, filename: str, payload: object) -> None:
        (self.session_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_jsonl(self, filename: str, rows: list[dict]) -> None:
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        (self.session_dir / filename).write_text(text + "\n", encoding="utf-8")


MockAgentLoop = CompetitionAgentLoop
