# Observation Driven Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Current Progress - 2026-07-06

Status: implementation complete and verified.

- Current git log includes `a38a073 feat: drive agent loop from llm actions`.
- `app/agent/actions.py` defines typed tool and terminal actions.
- `ToolRegistry.snapshot()` is implemented and covered.
- `PlaceholderLLMProvider` produces the deterministic action sequence
  `plan -> category_insight -> item_search -> shipping -> rank -> pick -> finish`.
- `CompetitionAgentLoop.run()` now asks the configured LLM provider for each
  action, emits thought/provider/tool/ranking/result events, handles
  `finish`, `clarify`, `fail`, and emits budget exhaustion via `max_steps`.
- Output persistence writes `summary.json`, `candidates.json`, and `trace.jsonl`.
- Verification rerun on 2026-07-06:
  - `uv run pytest -q` -> `48 passed, 1 warning`
  - `OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py` -> exits 0

Remaining follow-up:

- Add `provider_calls.jsonl` if separate provider-call audit output is still
  required by the competition trace contract.
- Expand dynamic-agent tests around malformed real LLM output and provider
  partial failures.

**Goal:** Replace the current fixed `plan -> category_insight -> item_search -> shipping -> rank -> pick` sequence with an observation-driven loop that asks the configured LLM provider for the next action, reacts to observations, and terminates with either a recommendation, clarification request, provider failure, or budget exhaustion.

**Architecture:** Keep existing provider, tool, API, frontend, ranking, and trace surfaces. Add a small action/state model beside `CompetitionAgentLoop`, normalize LLM action proposals before tool execution, then make `CompetitionAgentLoop.run()` iterate until a terminal action or budget limit is reached. Tests use deterministic fake providers so `test` and `submission` remain offline.

**Tech Stack:** Python 3.10, uv, pytest, pytest-asyncio, Pydantic, dataclasses, existing FastAPI task state and provider contracts.

## Global Constraints

- Do not change profile defaults, provider registry behavior, real adapter auth, frontend UI, or ranking score formulas in this plan.
- `OMNIMATCH_PROFILE=submission` must continue to run without secrets using deterministic placeholder providers.
- `OMNIMATCH_PROFILE=test` must not call network APIs.
- Existing tool names stay compatible: `plan`, `category_insight`, `item_search`, `shipping`, `rank`, `pick`.
- New terminal actions are `finish`, `clarify`, and `fail`.
- The loop must emit existing event types where possible: `task_started`, `thought`, `tool_start`, `tool_end`, `provider_start`, `provider_end`, `ranking_decision`, `task_result`, `task_error`.
- Output files remain `summary.json`, `candidates.json`, and `trace.jsonl`.
- This plan describes future implementation work. The current turn only adds this documentation file.

---

## Current State Summary

- `app/config.py` already defines `dev`, `submission`, and `test` profiles.
- `app/providers/registry.py` already selects placeholder, SerpApi, Serper, OpenAI-compatible, HTTP product, HTTP web search, and shipping adapters.
- `app/providers/base.py` already exposes `LLMProvider.plan_next_action(messages)`.
- `app/agent/tool_registry.py` already owns tool state: intent, insight, candidates, and scored candidates.
- `app/agent/main_agent.py` currently ignores `LLMProvider.plan_next_action()` after initial intent extraction and runs a fixed hard-coded action list.
- Existing tests assert provider/ranking events and trace files, but they do not verify action choice, branching, clarification, or budget exhaustion.

## File Structure

- Create: `app/agent/actions.py`
  - Owns `AgentAction`, `AgentStep`, action normalization, allowed action names, and malformed-action handling.
- Modify: `app/agent/main_agent.py`
  - Replaces fixed sequence with a bounded observation-driven loop.
  - Builds LLM decision messages from query, completed actions, observations, and current tool state.
  - Handles terminal actions and output persistence.
- Modify: `app/agent/tool_registry.py`
  - Adds `snapshot()` so the loop can expose concise state to the LLM and trace without reaching into every field ad hoc.
- Modify: `app/providers/placeholder.py`
  - Makes `PlaceholderLLMProvider.plan_next_action()` deterministic across repeated loop calls.
- Modify: `app/tools/shopping_summary.py`
  - Adds an optional terminal note for clarification, failure, and budget exhaustion summaries.
- Modify: `app/schemas.py`
  - Adds optional `status_note` and `uncertainty` fields to `ShoppingSummary` while preserving existing required response fields.
- Modify: `tests/test_agent_loop.py`
  - Adds deterministic fake-LLM tests for dynamic branching, clarification, and budget exhaustion.
- Modify: `tests/test_tools.py`
  - Adds `ToolRegistry.snapshot()` contract coverage.

---

### Task 1: Add Typed Agent Actions

**Files:**
- Create: `app/agent/actions.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Produces: `ToolActionName = Literal["plan", "category_insight", "item_search", "shipping", "rank", "pick"]`
- Produces: `TerminalActionName = Literal["finish", "clarify", "fail"]`
- Produces: `AgentAction.from_provider_data(data: dict[str, Any]) -> AgentAction`
- Produces: `AgentStep(action: AgentAction, observation_count: int, observations: list[dict[str, Any]])`

- [ ] **Step 1: Write failing action normalization tests**

Add these tests to `tests/test_agent_loop.py`:

```python
from app.agent.actions import AgentAction


def test_agent_action_normalizes_tool_action():
    action = AgentAction.from_provider_data(
        {
            "action": "item_search",
            "arguments": {"platforms": ["Amazon"]},
            "thought": "Need candidates before ranking.",
        }
    )

    assert action.name == "item_search"
    assert action.arguments == {"platforms": ["Amazon"]}
    assert action.thought == "Need candidates before ranking."
    assert action.is_terminal is False


def test_agent_action_turns_unknown_action_into_fail():
    action = AgentAction.from_provider_data({"action": "delete_everything", "arguments": []})

    assert action.name == "fail"
    assert action.is_terminal is True
    assert "unknown action" in action.message
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_agent_action_normalizes_tool_action tests/test_agent_loop.py::test_agent_action_turns_unknown_action_into_fail -q
```

Expected:

```text
ModuleNotFoundError: No module named 'app.agent.actions'
```

- [ ] **Step 3: Implement `app/agent/actions.py`**

Create `app/agent/actions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ToolActionName = Literal["plan", "category_insight", "item_search", "shipping", "rank", "pick"]
TerminalActionName = Literal["finish", "clarify", "fail"]
ActionName = ToolActionName | TerminalActionName

TOOL_ACTIONS: set[str] = {"plan", "category_insight", "item_search", "shipping", "rank", "pick"}
TERMINAL_ACTIONS: set[str] = {"finish", "clarify", "fail"}


@dataclass(frozen=True)
class AgentAction:
    name: ActionName
    arguments: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    message: str = ""

    @classmethod
    def from_provider_data(cls, data: dict[str, Any]) -> "AgentAction":
        raw_name = str(data.get("action") or "").strip()
        raw_arguments = data.get("arguments")
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
        thought = str(data.get("thought") or "")
        message = str(data.get("message") or "")

        if raw_name in TOOL_ACTIONS or raw_name in TERMINAL_ACTIONS:
            return cls(
                name=raw_name,  # type: ignore[arg-type]
                arguments=arguments,
                thought=thought,
                message=message,
            )

        return cls(
            name="fail",
            arguments={},
            thought=thought,
            message=f"unknown action from LLM provider: {raw_name or '<missing>'}",
        )

    @property
    def is_terminal(self) -> bool:
        return self.name in TERMINAL_ACTIONS


@dataclass(frozen=True)
class AgentStep:
    action: AgentAction
    observation_count: int
    observations: list[dict[str, Any]]
```

- [ ] **Step 4: Run action tests**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_agent_action_normalizes_tool_action tests/test_agent_loop.py::test_agent_action_turns_unknown_action_into_fail -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add app/agent/actions.py tests/test_agent_loop.py
git commit -m "feat: add typed agent actions"
```

---

### Task 2: Expose Tool Registry State Snapshots

**Files:**
- Modify: `app/agent/tool_registry.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `ToolRegistry.run(action: str, arguments: dict[str, Any]) -> object`
- Produces: `ToolRegistry.snapshot() -> dict[str, Any]`

- [ ] **Step 1: Write failing snapshot test**

Add this import to `tests/test_tools.py`:

```python
from app.agent.tool_registry import ToolRegistry
```

Add this test:

```python
@pytest.mark.asyncio
async def test_tool_registry_snapshot_reports_progress():
    settings = OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="placeholder",
    )
    ctx = ToolContext(settings=settings, providers=ProviderRegistry.from_settings(settings))
    registry = ToolRegistry(ctx)

    initial = registry.snapshot()
    assert initial == {
        "has_intent": False,
        "has_insight": False,
        "candidate_count": 0,
        "scored_count": 0,
        "top_score": None,
    }

    await registry.run("plan", {"query": "旅行三件套，预算300，不要塑料"})
    await registry.run("category_insight", {})
    await registry.run("item_search", {})
    await registry.run("shipping", {})
    await registry.run("rank", {})

    after_rank = registry.snapshot()
    assert after_rank["has_intent"] is True
    assert after_rank["has_insight"] is True
    assert after_rank["candidate_count"] == 4
    assert after_rank["scored_count"] == 4
    assert after_rank["top_score"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_tools.py::test_tool_registry_snapshot_reports_progress -q
```

Expected:

```text
AttributeError: 'ToolRegistry' object has no attribute 'snapshot'
```

- [ ] **Step 3: Implement `ToolRegistry.snapshot()`**

Add this method to `app/agent/tool_registry.py` inside `ToolRegistry`:

```python
    def snapshot(self) -> dict[str, Any]:
        return {
            "has_intent": self.intent is not None,
            "has_insight": self.insight is not None,
            "candidate_count": len(self.candidates),
            "scored_count": len(self.scored),
            "top_score": self.scored[0].score.total if self.scored else None,
        }
```

- [ ] **Step 4: Run snapshot test**

Run:

```bash
uv run pytest tests/test_tools.py::test_tool_registry_snapshot_reports_progress -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add app/agent/tool_registry.py tests/test_tools.py
git commit -m "feat: expose agent tool progress snapshots"
```

---

### Task 3: Make Placeholder LLM Drive A Deterministic Loop

**Files:**
- Modify: `app/providers/placeholder.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `PlaceholderLLMProvider.plan_next_action(messages: list[dict[str, Any]])`
- Produces: deterministic action sequence: `plan`, `category_insight`, `item_search`, `shipping`, `rank`, `pick`, `finish`

- [ ] **Step 1: Write failing placeholder sequence test**

Add this test to `tests/test_providers.py`:

```python
@pytest.mark.asyncio
async def test_placeholder_llm_proposes_deterministic_action_sequence():
    provider = PlaceholderLLMProvider()
    actions: list[str] = []

    for _ in range(7):
        result = await provider.plan_next_action([{"role": "user", "content": "next"}])
        actions.append(result.data["action"])

    assert actions == [
        "plan",
        "category_insight",
        "item_search",
        "shipping",
        "rank",
        "pick",
        "finish",
    ]
```

Also add this import:

```python
from app.providers.placeholder import PlaceholderLLMProvider
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_providers.py::test_placeholder_llm_proposes_deterministic_action_sequence -q
```

Expected:

```text
AssertionError
```

- [ ] **Step 3: Implement deterministic sequence**

Replace `PlaceholderLLMProvider` in `app/providers/placeholder.py` with:

```python
class PlaceholderLLMProvider:
    provider = "placeholder_llm"

    def __init__(self) -> None:
        self._index = 0
        self._actions = [
            "plan",
            "category_insight",
            "item_search",
            "shipping",
            "rank",
            "pick",
            "finish",
        ]

    async def plan_next_action(self, messages: list[dict[str, Any]]) -> ProviderResult[dict[str, Any]]:
        start = time.perf_counter()
        action = self._actions[min(self._index, len(self._actions) - 1)]
        self._index += 1
        return ProviderResult(
            provider=self.provider,
            provider_mode="placeholder",
            latency_ms=_latency_ms(start),
            data={
                "action": action,
                "arguments": {},
                "thought": f"placeholder selected {action}",
            },
            warnings=["placeholder LLM used"],
            response_summary=f"deterministic placeholder action={action}",
        )
```

- [ ] **Step 4: Run placeholder provider tests**

Run:

```bash
uv run pytest tests/test_providers.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 5: Commit**

```bash
git add app/providers/placeholder.py tests/test_providers.py
git commit -m "feat: make placeholder llm drive loop actions"
```

---

### Task 4: Replace Fixed Agent Sequence With Action Loop

**Files:**
- Modify: `app/agent/main_agent.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `AgentAction.from_provider_data(data)`
- Consumes: `ToolRegistry.snapshot()`
- Produces: `CompetitionAgentLoop(..., max_steps: int = 8)`
- Produces: `thought` event for every provider-proposed action with `payload.action`

- [ ] **Step 1: Write fake LLM provider and dynamic sequence test**

Add this helper to `tests/test_agent_loop.py`:

```python
from app.providers.base import ProviderResult


class SequenceLLMProvider:
    def __init__(self, actions: list[dict]) -> None:
        self.actions = actions
        self.calls: list[list[dict]] = []

    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        self.calls.append(messages)
        index = min(len(self.calls) - 1, len(self.actions) - 1)
        return ProviderResult(
            provider="sequence_llm",
            provider_mode="fake",
            latency_ms=1,
            data=self.actions[index],
            response_summary=f"sequence action {index}",
        )
```

Add this test:

```python
@pytest.mark.asyncio
async def test_competition_loop_uses_llm_action_sequence(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    llm = SequenceLLMProvider(
        [
            {"action": "plan", "arguments": {}, "thought": "Extract constraints."},
            {"action": "item_search", "arguments": {}, "thought": "Skip category insight."},
            {"action": "rank", "arguments": {}, "thought": "Rank raw candidates."},
            {"action": "pick", "arguments": {}, "thought": "Pick winners."},
            {"action": "finish", "message": "Enough evidence."},
        ]
    )
    providers = ProviderRegistry(
        llm=llm,
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_dynamic")
    loop = CompetitionAgentLoop(
        thread_id="thread_dynamic",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    summary = await loop.run("旅行三件套，预算300，不要塑料")

    tool_starts = [event.tool for event in monitor.events if event.type == "tool_start"]
    thoughts = [event for event in monitor.events if event.type == "thought"]
    assert tool_starts == ["plan", "item_search", "rank", "pick"]
    assert [event.payload["action"] for event in thoughts] == [
        "plan",
        "item_search",
        "rank",
        "pick",
        "finish",
    ]
    assert len(llm.calls) == 5
    assert summary.products
```

- [ ] **Step 2: Run test to verify it fails against fixed sequence**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_competition_loop_uses_llm_action_sequence -q
```

Expected:

```text
AssertionError
```

The current implementation emits fixed `tool_start` events including `category_insight` and `shipping`.

- [ ] **Step 3: Implement action loop**

Replace `CompetitionAgentLoop` in `app/agent/main_agent.py` with this structure, preserving imports not shown if still used:

```python
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
            action = await self._plan_next_action(query, step_index, tools, ctx, steps)
            await self.monitor.emit(
                "thought",
                action.thought or action.message or f"Selected action {action.name}.",
                payload={"action": action.name, "arguments": action.arguments},
            )

            if action.name == "finish":
                terminal_note = action.message
                break
            if action.name == "clarify":
                terminal_note = action.message or "需要更多信息才能给出可靠推荐。"
                break
            if action.name == "fail":
                terminal_note = action.message or "代理无法继续执行。"
                await self.monitor.emit("task_error", terminal_note)
                break

            await self.monitor.emit("tool_start", f"{action.name} started", tool=action.name)
            observation_start = len(ctx.observations)
            result = await tools.run(action.name, self._tool_arguments(action, query))
            new_observations = ctx.observations[observation_start:]
            steps.append(
                AgentStep(
                    action=action,
                    observation_count=len(ctx.observations),
                    observations=new_observations,
                )
            )
            await self._emit_provider_events(action.name, new_observations)
            await self.monitor.emit("tool_end", f"{action.name} finished", tool=action.name)

            if action.name == "rank":
                await self.monitor.emit(
                    "ranking_decision",
                    "Candidates scored.",
                    payload={"candidate_count": len(result)},
                )
            if action.name == "pick":
                picked = result
        else:
            terminal_note = f"Reached max_steps={self.max_steps} before a finish action."
            await self.monitor.emit("task_error", terminal_note)

        summary = await build_summary(query, picked or [], ctx, status_note=terminal_note)
        try:
            self._write_json("summary.json", summary.model_dump())
            self._write_json("candidates.json", [item.model_dump() for item in tools.scored])
            self._write_jsonl("trace.jsonl", [self._step_to_trace(step) for step in steps])
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
    ) -> AgentAction:
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
        ctx.observations.append(
            {
                "tool": "AgentPlanner",
                "provider": result.provider,
                "provider_mode": result.provider_mode,
                "latency_ms": result.latency_ms,
                "warnings": result.warnings,
            }
        )
        return AgentAction.from_provider_data(result.data)

    def _tool_arguments(self, action: AgentAction, query: str) -> dict[str, Any]:
        if action.name == "plan":
            return {"query": action.arguments.get("query") or query}
        return action.arguments

    async def _emit_provider_events(self, action_name: str, observations: list[dict[str, Any]]) -> None:
        for observation in observations:
            if observation.get("provider"):
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

    def _step_to_trace(self, step: AgentStep) -> dict[str, Any]:
        return {
            "action": step.action.name,
            "thought": step.action.thought,
            "message": step.action.message,
            "observation_count": step.observation_count,
            "observations": step.observations,
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
```

- [ ] **Step 4: Update `build_summary()` signature**

Modify `app/tools/shopping_summary.py` function signature and return block:

```python
async def build_summary(
    query: str,
    picked: list[ScoredProduct],
    ctx: ToolContext,
    status_note: str = "",
) -> ShoppingSummary:
```

Set the message with:

```python
    if status_note and not products:
        message = f"基于“{query}”，{status_note}"
    else:
        message = f"基于“{query}”，为你推荐 {count} 件商品，已按约束、证据和含运费总价排序。"
```

Then return:

```python
    return ShoppingSummary(
        message=message,
        products=products,
        warnings=[f"evidence used provider modes: {provider_modes}"] if provider_modes else [],
        status_note=status_note,
    )
```

- [ ] **Step 5: Add optional schema fields**

Modify `ShoppingSummary` in `app/schemas.py`:

```python
class ShoppingSummary(BaseModel):
    message: str
    products: list[Product]
    warnings: list[str] = Field(default_factory=list)
    status_note: str = ""
    uncertainty: list[str] = Field(default_factory=list)
```

- [ ] **Step 6: Run dynamic loop test**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_competition_loop_uses_llm_action_sequence -q
```

Expected:

```text
1 passed
```

- [ ] **Step 7: Run existing agent/tool/schema tests**

Run:

```bash
uv run pytest tests/test_agent_loop.py tests/test_tools.py tests/test_schemas.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 8: Commit**

```bash
git add app/agent/main_agent.py app/schemas.py app/tools/shopping_summary.py tests/test_agent_loop.py
git commit -m "feat: drive agent loop from llm actions"
```

---

### Task 5: Add Clarification And Budget Terminals

**Files:**
- Modify: `app/agent/main_agent.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: terminal action `clarify`
- Consumes: `CompetitionAgentLoop(max_steps=...)`
- Produces: summary with no products and `status_note` when clarification or step budget stops the loop.

- [ ] **Step 1: Write clarification terminal test**

Add this test to `tests/test_agent_loop.py`:

```python
@pytest.mark.asyncio
async def test_competition_loop_can_request_clarification(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    llm = SequenceLLMProvider(
        [
            {
                "action": "clarify",
                "thought": "The request is too broad.",
                "message": "请补充预算和商品类别。",
            }
        ]
    )
    providers = ProviderRegistry(
        llm=llm,
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_clarify")
    loop = CompetitionAgentLoop(
        thread_id="thread_clarify",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    summary = await loop.run("帮我买点东西")

    assert summary.products == []
    assert summary.status_note == "请补充预算和商品类别。"
    assert "请补充预算和商品类别" in summary.message
    assert [event.type for event in monitor.events] == ["task_started", "thought", "task_result"]
```

- [ ] **Step 2: Write budget exhaustion test**

Add this test to `tests/test_agent_loop.py`:

```python
@pytest.mark.asyncio
async def test_competition_loop_stops_at_max_steps(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    llm = SequenceLLMProvider(
        [
            {"action": "plan", "arguments": {}, "thought": "Plan again."},
            {"action": "plan", "arguments": {}, "thought": "Plan again."},
            {"action": "plan", "arguments": {}, "thought": "Plan again."},
        ]
    )
    providers = ProviderRegistry(
        llm=llm,
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_budget")
    loop = CompetitionAgentLoop(
        thread_id="thread_budget",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
        max_steps=2,
    )

    summary = await loop.run("旅行三件套")

    assert summary.products == []
    assert summary.status_note == "Reached max_steps=2 before a finish action."
    assert "Reached max_steps=2" in summary.message
    assert "task_error" in [event.type for event in monitor.events]
```

- [ ] **Step 3: Run tests**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_competition_loop_can_request_clarification tests/test_agent_loop.py::test_competition_loop_stops_at_max_steps -q
```

Expected:

```text
2 passed
```

- [ ] **Step 4: Commit**

```bash
git add app/agent/main_agent.py tests/test_agent_loop.py
git commit -m "feat: add agent terminal conditions"
```

---

### Task 6: Persist Action Planning Observability

**Files:**
- Modify: `app/agent/main_agent.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `AgentPlanner` observations added by `_plan_next_action()`
- Produces: `trace.jsonl` rows containing action, thought, message, and observations.
- Produces: provider events for `AgentPlanner` observations as well as tool observations.

- [ ] **Step 1: Write trace observability test**

Add this test to `tests/test_agent_loop.py`:

```python
import json


@pytest.mark.asyncio
async def test_competition_loop_trace_records_action_thoughts(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    llm = SequenceLLMProvider(
        [
            {"action": "plan", "arguments": {}, "thought": "Understand the request."},
            {"action": "finish", "message": "Done for test."},
        ]
    )
    providers = ProviderRegistry(
        llm=llm,
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_trace")
    loop = CompetitionAgentLoop(
        thread_id="thread_trace",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("旅行三件套")

    rows = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["action"] == "plan"
    assert rows[0]["thought"] == "Understand the request."
    assert any(observation["tool"] == "Planner" for observation in rows[0]["observations"])
    provider_events = [event for event in monitor.events if event.type == "provider_end"]
    assert any(event.payload.get("tool") == "AgentPlanner" for event in provider_events)
```

- [ ] **Step 2: Run test to verify current gap**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_competition_loop_trace_records_action_thoughts -q
```

Expected:

```text
AssertionError
```

This fails because `_plan_next_action()` observations are not included in the per-step observation slice.

- [ ] **Step 3: Include planning observations in step trace and provider events**

In `app/agent/main_agent.py`, update `_plan_next_action()` to return both the action and planner observation:

```python
    async def _plan_next_action(...) -> tuple[AgentAction, dict[str, Any]]:
        ...
        planner_observation = {
            "tool": "AgentPlanner",
            "provider": result.provider,
            "provider_mode": result.provider_mode,
            "latency_ms": result.latency_ms,
            "warnings": result.warnings,
        }
        ctx.observations.append(planner_observation)
        return AgentAction.from_provider_data(result.data), planner_observation
```

In `run()`, unpack and emit the planner provider event before the thought event:

```python
            action, planner_observation = await self._plan_next_action(
                query,
                step_index,
                tools,
                ctx,
                steps,
            )
            await self._emit_provider_events(action.name, [planner_observation])
```

When recording a tool step, include the planner observation in the row:

```python
            step_observations = [planner_observation, *new_observations]
            steps.append(
                AgentStep(
                    action=action,
                    observation_count=len(ctx.observations),
                    observations=step_observations,
                )
            )
```

For terminal actions, append a terminal step before breaking:

```python
            if action.is_terminal:
                steps.append(
                    AgentStep(
                        action=action,
                        observation_count=len(ctx.observations),
                        observations=[planner_observation],
                    )
                )
```

- [ ] **Step 4: Run trace observability test**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_competition_loop_trace_records_action_thoughts -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Run all agent tests**

Run:

```bash
uv run pytest tests/test_agent_loop.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 6: Commit**

```bash
git add app/agent/main_agent.py tests/test_agent_loop.py
git commit -m "feat: persist agent planning trace"
```

---

### Task 7: Verification And Submission Smoke

**Files:**
- Modify: none unless verification exposes regressions.

**Interfaces:**
- Consumes: all previous task interfaces.
- Produces: verified observation-driven loop that still runs offline in `submission`.

- [ ] **Step 1: Run backend tests**

Run:

```bash
uv run pytest -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 2: Run no-secret submission smoke**

Run:

```bash
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
```

Expected:

```text
The command exits 0, writes output/thread_*/summary.json, output/thread_*/candidates.json, and output/thread_*/trace.jsonl, and the summary warnings mention provider modes: placeholder.
```

- [ ] **Step 3: Inspect latest trace shape**

Run:

```bash
ls -td output/thread_* | head -1
```

Then replace `<latest>` below with the printed directory:

```bash
sed -n '1,5p' <latest>/trace.jsonl
```

Expected:

```text
Each JSON line contains action, thought, message, observation_count, and observations.
At least one observation has "tool": "AgentPlanner".
```

- [ ] **Step 4: Commit final fixes if needed**

If verification required fixes, commit them:

```bash
git add app tests
git commit -m "fix: stabilize observation driven agent loop"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review

- Spec coverage: This plan addresses the remaining gap from `docs/superpowers/specs/2026-07-03-competition-agent-design.md`: the loop reacts to provider-proposed actions instead of executing only a fixed sequence, and it covers clarification and max-step terminal conditions.
- Placeholder scan: The plan contains no unresolved placeholder markers or unspecified implementation steps.
- Type consistency: `AgentAction`, `AgentStep`, `ToolRegistry.snapshot()`, `CompetitionAgentLoop(max_steps=...)`, and `ShoppingSummary.status_note` are defined before they are consumed by later tasks.
- Scope control: The plan intentionally leaves provider adapters, ranking formulas, frontend rendering, and config behavior unchanged.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-03-observation-driven-agent-loop-implementation.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.
