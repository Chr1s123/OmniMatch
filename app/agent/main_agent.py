from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agent.actions import AgentAction, AgentStep
from app.agent.forking import (
    AgentScope,
    ForkExecutor,
    ForkRequest,
    SubAgentPayload,
    SubAgentResult,
    SubAgentPayloadStatus,
    sanitize_subagent_error,
    thaw_context_snapshot,
)
from app.agent.tool_registry import ToolRegistry
from app.api.context import set_task_context
from app.api.monitor import EventEmitter, ScopedEventCollector
from app.config import OmniMatchSettings
from app.providers.registry import ProviderRegistry
from app.schemas import ShoppingSummary
from app.tools.context import ToolContext
from app.tools.shopping_summary import build_summary


_ROOT_TOOL_ACTIONS = ("plan", "category_insight", "item_search", "shipping", "rank", "pick")


class CompetitionAgentLoop:
    def __init__(
        self,
        thread_id: str,
        session_dir: str | Path,
        settings: OmniMatchSettings,
        providers: ProviderRegistry,
        monitor: EventEmitter,
        max_steps: int = 8,
        scope: AgentScope | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.session_dir = Path(session_dir)
        self.settings = settings
        self.providers = providers
        self.monitor = monitor
        self.max_steps = max_steps
        self.scope = scope or AgentScope()
        self.last_observations: list[dict[str, Any]] = []
        self.last_step_count = 0
        self.last_status: SubAgentPayloadStatus = "completed"
        self.last_error: str | None = None

    async def run(self, query: str) -> ShoppingSummary:
        set_task_context(self.thread_id, self.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(settings=self.settings, providers=self.providers)
        tools = ToolRegistry(ctx, allowed_tools=self.scope.allowed_tools)
        trace: list[dict[str, Any]] = []
        steps: list[AgentStep] = []
        picked = None
        terminal_note = ""
        self.last_status = "completed"
        self.last_error = None

        if self.scope.emit_task_result:
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
            action = self._ensure_executable_action(action, tools, query)
            action = self._sanitize_child_terminal_action(action)
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
                terminal_note = await self._emit_terminal_error(terminal_note)
                steps.append(
                    AgentStep(
                        action=action,
                        observation_count=len(ctx.observations),
                        observations=[planner_observation],
                    )
                )
                trace.append(self._trace_row(action, [planner_observation], len(ctx.observations)))
                break

            if action.name == "fork":
                results = await self._execute_fork(action)
                observation = {
                    "tool": "fork",
                    "subagents": [result.model_dump() for result in results],
                }
                ctx.observations.append(observation)
                step_observations = [planner_observation, observation]
                steps.append(
                    AgentStep(
                        action=action,
                        observation_count=len(ctx.observations),
                        observations=step_observations,
                    )
                )
                trace.append(self._trace_row(action, step_observations, len(ctx.observations)))
                continue

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
            terminal_note = await self._emit_terminal_error(terminal_note)

        self.last_observations = list(ctx.observations)
        self.last_step_count = len(steps)
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
        if self.scope.emit_task_result:
            await self.monitor.emit(
                "task_result",
                "Shopping summary generated.",
                payload={"summary": summary.model_dump()},
            )
        return summary

    async def _execute_fork(self, action: AgentAction) -> list[SubAgentResult]:
        requests = ForkRequest.parse_many(action.arguments, self.settings)
        child_depth = self.scope.depth + 1
        if child_depth > self.settings.max_fork_depth:
            return sorted(
                [
                    SubAgentResult(
                        task_id=request.task_id,
                        status="failed",
                        error=(
                            f"fork depth {child_depth} exceeds "
                            f"max_fork_depth={self.settings.max_fork_depth}"
                        ),
                    )
                    for request in requests
                ],
                key=lambda result: result.task_id,
            )

        root_monitor = (
            self.monitor.parent if isinstance(self.monitor, ScopedEventCollector) else self.monitor
        )
        executor = ForkExecutor(
            monitor=ScopedEventCollector(root_monitor, {"fork_depth": child_depth}),
            max_parallel=self.settings.max_parallel_subagents,
        )

        async def run_child(request: ForkRequest) -> SubAgentPayload:
            child = CompetitionAgentLoop(
                thread_id=self.thread_id,
                session_dir=self.session_dir / "subagents" / request.task_id,
                settings=self.settings,
                providers=self.providers,
                monitor=ScopedEventCollector(
                    root_monitor,
                    {
                        "subagent_id": request.task_id,
                        "fork_depth": child_depth,
                    },
                ),
                max_steps=request.max_steps,
                scope=AgentScope(
                    depth=child_depth,
                    task_id=request.task_id,
                    allowed_tools=frozenset(request.allowed_tools),
                    emit_task_result=False,
                    context_snapshot=request.context_snapshot,
                ),
            )
            summary = await child.run(request.objective)
            return SubAgentPayload(
                status=child.last_status,
                result={
                    "merge_key": request.merge_key,
                    "summary": summary.model_dump(),
                },
                observations=child.last_observations,
                warnings=summary.warnings,
                error=child.last_error,
                step_count=child.last_step_count,
            )

        return await executor.execute(requests, run_child)

    async def _plan_next_action(
        self,
        query: str,
        step_index: int,
        tools: ToolRegistry,
        ctx: ToolContext,
        steps: list[AgentStep],
    ) -> tuple[AgentAction, dict[str, Any]]:
        allowed_tool_actions = (
            _ROOT_TOOL_ACTIONS
            if self.scope.allowed_tools is None
            else tuple(sorted(self.scope.allowed_tools))
        )
        fork_instruction = (
            "Allowed orchestration action: fork. "
            if self.scope.depth < self.settings.max_fork_depth
            else "Fork is not allowed at this depth. "
        )
        result = await self.providers.llm.plan_next_action(
            [
                {
                    "role": "system",
                    "content": (
                        "Choose the next shopping-agent action as JSON. Allowed tool actions: "
                        f"{', '.join(allowed_tool_actions)}. "
                        f"{fork_instruction}"
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
                            "fork_depth": self.scope.depth,
                            "context_snapshot": thaw_context_snapshot(self.scope.context_snapshot),
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

    def _ensure_executable_action(
        self,
        action: AgentAction,
        tools: ToolRegistry,
        query: str,
    ) -> AgentAction:
        if (
            action.is_terminal
            or action.is_orchestration
            or action.name == "plan"
            or tools.snapshot()["has_intent"]
        ):
            return action
        return AgentAction(
            name="plan",
            arguments={"query": query},
            thought=f"Planner state is required before {action.name}; running plan first.",
            message=action.message,
        )

    def _sanitize_child_terminal_action(self, action: AgentAction) -> AgentAction:
        if self.scope.depth == 0 or action.name != "fail":
            return action
        return AgentAction(
            name="fail",
            arguments={},
            thought=sanitize_subagent_error(action.thought),
            message=sanitize_subagent_error(action.message),
        )

    async def _emit_terminal_error(self, terminal_note: str) -> str:
        self.last_status = "failed"
        if self.scope.depth > 0:
            terminal_note = sanitize_subagent_error(terminal_note)
            self.last_error = terminal_note
            await self.monitor.emit("subagent_error", terminal_note)
        else:
            self.last_error = terminal_note
            await self.monitor.emit("task_error", terminal_note)
        return terminal_note

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
