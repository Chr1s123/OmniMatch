# Homogeneous AgentLoop Fork Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unused platform-level `asyncio.gather` prototype with bounded, observable forks that run the same `CompetitionAgentLoop` implementation as the parent.

**Architecture:** Add typed fork requests/results and an orchestration action beside the existing tool and terminal actions. A `ForkExecutor` runs child loops with isolated `ToolRegistry` state, scoped events, stable result ordering, depth/concurrency/step/time budgets, and cancellation propagation; only the root loop emits the user-facing `task_result`.

**Tech Stack:** Python 3.10, asyncio, Pydantic 2, existing provider interfaces, pytest, pytest-asyncio.

## Global Constraints

- The main agent paradigm remains `Think -> Act -> Observe -> Reflect`.
- Multi-agent coordination uses homogeneous `CompetitionAgentLoop` forks only; do not introduce role-specific agents or another agent framework.
- Only the root loop may emit the final user-facing `task_result`.
- Defaults are `max_fork_depth=1`, `max_parallel_subagents=4`, `subagent_max_steps=4`, and `subagent_timeout_seconds=30`.
- Child loops use isolated mutable state and a read-only context snapshot.
- Parent cancellation, timeout, or budget exhaustion cancels unfinished children.
- `test` and `submission` remain deterministic and network-free.
- `app/agent/dispatch_tool.py` is currently unused and is not a true sub-AgentLoop implementation.
- This plan does not implement Faiss, OpenSearch, vector models, or training.

---

## Current State

- `CompetitionAgentLoop` already asks the LLM for structured actions and stops on terminal actions or `max_steps`.
- `AgentAction` currently recognizes six tool actions and three terminal actions; it does not recognize `fork`.
- `ToolRegistry` owns mutable intent, insight, candidate, and score state but has no tool allowlist.
- `EventCollector` has no child scope enrichment.
- `dispatch_platform_search()` runs `search_items()` and `calculate_shipping()` with `asyncio.gather`, but no production code imports it.
- Existing `subagent_started` and `subagent_finished` events therefore demonstrate only a historical mock.

## File Structure

- Create: `app/agent/forking.py`
  - Owns `AgentScope`, `ForkRequest`, `SubAgentResult`, parsing, budget validation, and `ForkExecutor`.
- Modify: `app/agent/actions.py`
  - Adds the `fork` orchestration action without treating it as a tool or terminal action.
- Modify: `app/agent/main_agent.py`
  - Executes fork actions, creates homogeneous child loops, records fork observations, and suppresses child final events.
- Modify: `app/agent/tool_registry.py`
  - Enforces child tool allowlists.
- Modify: `app/api/monitor.py`
  - Adds an event-emitter protocol and scoped event wrapper.
- Modify: `app/config.py`
  - Adds typed fork limits and environment parsing.
- Modify: `.env.example`
  - Documents fork settings.
- Delete: `app/agent/dispatch_tool.py`
  - Removes the unused function-level concurrency prototype after the real fork path is covered.
- Modify: `tests/test_config.py`
  - Covers defaults, environment overrides, and invalid limits.
- Modify: `tests/test_agent_loop.py`
  - Covers action parsing, homogeneous execution, stable merge, depth limits, partial failures, and root-only results.
- Modify: `tests/test_tools.py`
  - Covers tool allowlists.

### Task 1: Add Typed Fork Budgets To Settings

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `OmniMatchSettings.max_fork_depth: int`
- Produces: `OmniMatchSettings.max_parallel_subagents: int`
- Produces: `OmniMatchSettings.subagent_max_steps: int`
- Produces: `OmniMatchSettings.subagent_timeout_seconds: float`

- [ ] **Step 1: Write failing configuration tests**

Add to `tests/test_config.py`:

```python
from dataclasses import replace


def submission_settings() -> OmniMatchSettings:
    return OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="placeholder",
    )


def test_fork_settings_have_bounded_defaults():
    settings = submission_settings()

    assert settings.max_fork_depth == 1
    assert settings.max_parallel_subagents == 4
    assert settings.subagent_max_steps == 4
    assert settings.subagent_timeout_seconds == 30.0


def test_fork_settings_read_environment(monkeypatch):
    monkeypatch.setenv("OMNIMATCH_PROFILE", "submission")
    monkeypatch.setenv("OMNIMATCH_MAX_FORK_DEPTH", "2")
    monkeypatch.setenv("OMNIMATCH_MAX_PARALLEL_SUBAGENTS", "3")
    monkeypatch.setenv("OMNIMATCH_SUBAGENT_MAX_STEPS", "5")
    monkeypatch.setenv("OMNIMATCH_SUBAGENT_TIMEOUT_SECONDS", "12.5")

    settings = OmniMatchSettings.from_env()

    assert settings.max_fork_depth == 2
    assert settings.max_parallel_subagents == 3
    assert settings.subagent_max_steps == 5
    assert settings.subagent_timeout_seconds == 12.5


def test_fork_settings_reject_non_positive_limits():
    settings = replace(submission_settings(), max_parallel_subagents=0)

    with pytest.raises(ConfigError, match="max_parallel_subagents"):
        settings.validate()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_config.py::test_fork_settings_have_bounded_defaults tests/test_config.py::test_fork_settings_read_environment tests/test_config.py::test_fork_settings_reject_non_positive_limits -q
```

Expected: FAIL because the four settings do not exist.

- [ ] **Step 3: Implement settings and parsers**

Add after the existing URL fields in `OmniMatchSettings`:

```python
    max_fork_depth: int = 1
    max_parallel_subagents: int = 4
    subagent_max_steps: int = 4
    subagent_timeout_seconds: float = 30.0
```

Add module helpers in `app/config.py`:

```python
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
```

At the start of `from_env()`, after validating `profile`, create this dictionary and pass it to all three `OmniMatchSettings` constructor branches with `**fork_limits`:

```python
        fork_limits = {
            "max_fork_depth": _env_int("OMNIMATCH_MAX_FORK_DEPTH", 1),
            "max_parallel_subagents": _env_int(
                "OMNIMATCH_MAX_PARALLEL_SUBAGENTS", 4
            ),
            "subagent_max_steps": _env_int("OMNIMATCH_SUBAGENT_MAX_STEPS", 4),
            "subagent_timeout_seconds": _env_float(
                "OMNIMATCH_SUBAGENT_TIMEOUT_SECONDS", 30.0
            ),
        }
```

Append to `validate()` before provider validation:

```python
        if self.max_fork_depth < 0:
            raise ConfigError("max_fork_depth must be >= 0")
        if self.max_parallel_subagents < 1:
            raise ConfigError("max_parallel_subagents must be >= 1")
        if self.subagent_max_steps < 1:
            raise ConfigError("subagent_max_steps must be >= 1")
        if self.subagent_timeout_seconds <= 0:
            raise ConfigError("subagent_timeout_seconds must be > 0")
```

Append to `.env.example`:

```dotenv
OMNIMATCH_MAX_FORK_DEPTH=1
OMNIMATCH_MAX_PARALLEL_SUBAGENTS=4
OMNIMATCH_SUBAGENT_MAX_STEPS=4
OMNIMATCH_SUBAGENT_TIMEOUT_SECONDS=30
```

- [ ] **Step 4: Run configuration tests**

Run:

```bash
uv run pytest tests/test_config.py -q
```

Expected: all configuration tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py .env.example
git commit -m "feat: configure homogeneous agent forks"
```

### Task 2: Define Fork Actions, Scopes, And Results

**Files:**
- Create: `app/agent/forking.py`
- Modify: `app/agent/actions.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Produces: `OrchestrationActionName = Literal["fork"]`
- Produces: `AgentScope(depth, task_id, allowed_tools, emit_task_result)`
- Produces: `ForkRequest.parse_many(arguments, settings) -> list[ForkRequest]`
- Produces: `SubAgentResult`

- [ ] **Step 1: Write failing model and action tests**

Add to `tests/test_agent_loop.py`:

```python
from app.agent.forking import AgentScope, ForkRequest


def test_agent_action_normalizes_fork_action():
    action = AgentAction.from_provider_data(
        {
            "action": "fork",
            "arguments": {
                "tasks": [
                    {
                        "task_id": "amazon",
                        "objective": "Search Amazon",
                        "allowed_tools": ["plan", "item_search"],
                        "context_snapshot": {"query": "carry-on"},
                        "merge_key": "products",
                    }
                ]
            },
        }
    )

    assert action.name == "fork"
    assert action.is_terminal is False
    assert action.is_orchestration is True


def test_fork_request_uses_settings_budgets():
    settings = submission_settings()
    requests = ForkRequest.parse_many(
        {
            "tasks": [
                {
                    "task_id": "amazon",
                    "objective": "Search Amazon",
                    "allowed_tools": ["plan", "item_search"],
                    "context_snapshot": {},
                    "merge_key": "products",
                }
            ]
        },
        settings,
    )

    assert requests[0].max_steps == 4
    assert requests[0].timeout_seconds == 30.0
    assert AgentScope().depth == 0


def test_fork_request_rejects_unknown_tools():
    with pytest.raises(ValueError, match="unknown allowed tool"):
        ForkRequest.parse_many(
            {
                "tasks": [
                    {
                        "task_id": "unsafe",
                        "objective": "Do unsafe work",
                        "allowed_tools": ["delete_everything"],
                        "context_snapshot": {},
                        "merge_key": "products",
                    }
                ]
            },
            submission_settings(),
        )


def test_agent_scope_context_snapshot_is_read_only():
    scope = AgentScope(context_snapshot={"platform": "Amazon"})

    with pytest.raises(TypeError):
        scope.context_snapshot["platform"] = "eBay"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_agent_action_normalizes_fork_action tests/test_agent_loop.py::test_fork_request_uses_settings_budgets tests/test_agent_loop.py::test_fork_request_rejects_unknown_tools tests/test_agent_loop.py::test_agent_scope_context_snapshot_is_read_only -q
```

Expected: FAIL because `forking.py` and the orchestration action do not exist.

- [ ] **Step 3: Add the orchestration action**

Update the aliases and sets in `app/agent/actions.py`:

```python
ToolActionName = Literal["plan", "category_insight", "item_search", "shipping", "rank", "pick"]
OrchestrationActionName = Literal["fork"]
TerminalActionName = Literal["finish", "clarify", "fail"]
ActionName = ToolActionName | OrchestrationActionName | TerminalActionName

TOOL_ACTIONS: set[str] = {"plan", "category_insight", "item_search", "shipping", "rank", "pick"}
ORCHESTRATION_ACTIONS: set[str] = {"fork"}
TERMINAL_ACTIONS: set[str] = {"finish", "clarify", "fail"}
```

In `AgentAction.from_provider_data()`, accept all three sets:

```python
        if raw_name in TOOL_ACTIONS | ORCHESTRATION_ACTIONS | TERMINAL_ACTIONS:
            return cls(
                name=raw_name,  # type: ignore[arg-type]
                arguments=arguments,
                thought=thought,
                message=message,
            )
```

Add this property:

```python
    @property
    def is_orchestration(self) -> bool:
        return self.name in ORCHESTRATION_ACTIONS
```

- [ ] **Step 4: Implement `app/agent/forking.py` models**

Create `app/agent/forking.py` with:

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, field_validator

from app.agent.actions import TOOL_ACTIONS
from app.config import OmniMatchSettings


SubAgentStatus = Literal["completed", "failed", "cancelled", "timed_out"]


@dataclass(frozen=True)
class AgentScope:
    depth: int = 0
    task_id: str | None = None
    allowed_tools: frozenset[str] | None = None
    emit_task_result: bool = True
    context_snapshot: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_snapshot",
            MappingProxyType(deepcopy(dict(self.context_snapshot))),
        )


class ForkRequest(BaseModel):
    task_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    objective: str = Field(min_length=1, max_length=1000)
    allowed_tools: list[str] = Field(min_length=1)
    context_snapshot: dict[str, Any]
    max_steps: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    merge_key: str = Field(min_length=1, max_length=64)

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - TOOL_ACTIONS)
        if unknown:
            raise ValueError(f"unknown allowed tool: {', '.join(unknown)}")
        return list(dict.fromkeys(value))

    @classmethod
    def parse_many(
        cls,
        arguments: dict[str, Any],
        settings: OmniMatchSettings,
    ) -> list["ForkRequest"]:
        raw_tasks = arguments.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("fork arguments.tasks must be a non-empty list")
        if len(raw_tasks) > settings.max_parallel_subagents:
            raise ValueError(
                "fork task count exceeds max_parallel_subagents="
                f"{settings.max_parallel_subagents}"
            )
        requests: list[ForkRequest] = []
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                raise ValueError("each fork task must be an object")
            requests.append(
                cls.model_validate(
                    {
                        **raw,
                        "max_steps": raw.get("max_steps", settings.subagent_max_steps),
                        "timeout_seconds": raw.get(
                            "timeout_seconds", settings.subagent_timeout_seconds
                        ),
                    }
                )
            )
        task_ids = [request.task_id for request in requests]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("fork task_id values must be unique")
        return requests


class SubAgentResult(BaseModel):
    task_id: str
    status: SubAgentStatus
    result: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    step_count: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)


class SubAgentPayload(BaseModel):
    result: dict[str, Any]
    observations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    step_count: int = Field(default=0, ge=0)
```

- [ ] **Step 5: Run model and action tests**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_agent_action_normalizes_fork_action tests/test_agent_loop.py::test_fork_request_uses_settings_budgets tests/test_agent_loop.py::test_fork_request_rejects_unknown_tools tests/test_agent_loop.py::test_agent_scope_context_snapshot_is_read_only -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add app/agent/actions.py app/agent/forking.py tests/test_agent_loop.py
git commit -m "feat: define homogeneous fork contracts"
```

### Task 3: Isolate Tools And Scope Child Events

**Files:**
- Modify: `app/agent/tool_registry.py`
- Modify: `app/api/monitor.py`
- Test: `tests/test_tools.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Produces: `ToolRegistry(ctx, allowed_tools: frozenset[str] | None = None)`
- Produces: `EventEmitter` protocol
- Produces: `ScopedEventCollector(parent, scope_payload)`

- [ ] **Step 1: Write failing isolation tests**

Add to `tests/test_tools.py`:

```python
def submission_settings() -> OmniMatchSettings:
    return OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="placeholder",
    )


@pytest.mark.asyncio
async def test_tool_registry_rejects_tools_outside_child_allowlist():
    settings = submission_settings()
    ctx = ToolContext(settings=settings, providers=ProviderRegistry.from_settings(settings))
    registry = ToolRegistry(ctx, allowed_tools=frozenset({"plan"}))

    await registry.run("plan", {"query": "旅行三件套"})

    with pytest.raises(PermissionError, match="item_search"):
        await registry.run("item_search", {})
```

Add to `tests/test_agent_loop.py`:

```python
from app.api.monitor import ScopedEventCollector


@pytest.mark.asyncio
async def test_scoped_event_collector_enriches_child_events():
    parent = EventCollector(thread_id="thread_scope")
    child = ScopedEventCollector(
        parent,
        {"subagent_id": "amazon", "fork_depth": 1},
    )

    await child.emit("tool_start", "search", tool="item_search", payload={"k": 100})

    assert parent.events[-1].payload == {
        "k": 100,
        "subagent_id": "amazon",
        "fork_depth": 1,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_tools.py::test_tool_registry_rejects_tools_outside_child_allowlist tests/test_agent_loop.py::test_scoped_event_collector_enriches_child_events -q
```

Expected: FAIL because neither allowlists nor scoped collectors exist.

- [ ] **Step 3: Enforce the Tool Registry allowlist**

Change the constructor and the start of `run()` in `app/agent/tool_registry.py`:

```python
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
```

Keep the existing action branches immediately after this check.

- [ ] **Step 4: Add scoped event emission**

Add imports and definitions to `app/api/monitor.py`:

```python
from typing import Protocol


class EventEmitter(Protocol):
    async def emit(
        self,
        event_type: str,
        message: str,
        tool: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        raise NotImplementedError


class ScopedEventCollector:
    def __init__(
        self,
        parent: EventCollector,
        scope_payload: dict[str, Any],
    ) -> None:
        self.parent = parent
        self.scope_payload = dict(scope_payload)

    @property
    def events(self) -> list[AgentEvent]:
        return self.parent.events

    async def emit(
        self,
        event_type: str,
        message: str,
        tool: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        enriched = {**(payload or {}), **self.scope_payload}
        return await self.parent.emit(event_type, message, tool=tool, payload=enriched)
```

- [ ] **Step 5: Run isolation tests**

Run:

```bash
uv run pytest tests/test_tools.py::test_tool_registry_rejects_tools_outside_child_allowlist tests/test_agent_loop.py::test_scoped_event_collector_enriches_child_events -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add app/agent/tool_registry.py app/api/monitor.py tests/test_tools.py tests/test_agent_loop.py
git commit -m "feat: isolate child tools and events"
```

### Task 4: Execute Forks With Stable Merge And Cancellation

**Files:**
- Modify: `app/agent/forking.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `ForkRequest`, `SubAgentResult`, `EventEmitter`
- Produces: `ForkExecutor.execute(requests, runner) -> list[SubAgentResult]`
- Produces: `SubAgentRunner = Callable[[ForkRequest], Awaitable[dict[str, Any]]]`

- [ ] **Step 1: Write failing executor tests**

Add to `tests/test_agent_loop.py`:

```python
import asyncio

from app.agent.forking import ForkExecutor, SubAgentPayload, SubAgentResult


@pytest.mark.asyncio
async def test_fork_executor_returns_stable_order_and_partial_failures():
    monitor = EventCollector(thread_id="thread_forks")
    executor = ForkExecutor(monitor=monitor, max_parallel=2)
    requests = ForkRequest.parse_many(
        {
            "tasks": [
                {
                    "task_id": "b",
                    "objective": "fail",
                    "allowed_tools": ["plan"],
                    "context_snapshot": {},
                    "merge_key": "products",
                },
                {
                    "task_id": "a",
                    "objective": "succeed",
                    "allowed_tools": ["plan"],
                    "context_snapshot": {},
                    "merge_key": "products",
                },
            ]
        },
        submission_settings(),
    )

    async def runner(request: ForkRequest) -> SubAgentPayload:
        if request.objective == "fail":
            raise RuntimeError("provider failed")
        return SubAgentPayload(
            result={"objective": request.objective},
            observations=[{"tool": "plan"}],
            warnings=[],
            step_count=1,
        )

    results = await executor.execute(requests, runner)

    assert [result.task_id for result in results] == ["a", "b"]
    assert [result.status for result in results] == ["completed", "failed"]
    assert results[1].error == "provider failed"


@pytest.mark.asyncio
async def test_fork_executor_marks_timeout():
    monitor = EventCollector(thread_id="thread_timeout")
    executor = ForkExecutor(monitor=monitor, max_parallel=1)
    request = ForkRequest(
        task_id="slow",
        objective="slow",
        allowed_tools=["plan"],
        context_snapshot={},
        max_steps=1,
        timeout_seconds=0.01,
        merge_key="products",
    )

    async def runner(_: ForkRequest) -> SubAgentPayload:
        await asyncio.sleep(1)
        return SubAgentPayload(result={})

    results = await executor.execute([request], runner)

    assert results[0].status == "timed_out"
    assert "timed out" in (results[0].error or "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_fork_executor_returns_stable_order_and_partial_failures tests/test_agent_loop.py::test_fork_executor_marks_timeout -q
```

Expected: FAIL because `ForkExecutor` does not exist.

- [ ] **Step 3: Implement `ForkExecutor`**

Append to `app/agent/forking.py`:

```python
import asyncio
from collections.abc import Awaitable, Callable
import re
from time import perf_counter

from app.api.monitor import EventEmitter


SubAgentRunner = Callable[[ForkRequest], Awaitable[SubAgentPayload]]


def _safe_error(exc: Exception) -> str:
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", str(exc))
    return re.sub(r"(?i)(api[_-]?key|password)(\s*[:=]\s*)\S+", r"\1\2[REDACTED]", text)[:500]


class ForkExecutor:
    def __init__(self, monitor: EventEmitter, max_parallel: int) -> None:
        self.monitor = monitor
        self._semaphore = asyncio.Semaphore(max_parallel)

    async def execute(
        self,
        requests: list[ForkRequest],
        runner: SubAgentRunner,
    ) -> list[SubAgentResult]:
        tasks = [asyncio.create_task(self._run_one(request, runner)) for request in requests]
        try:
            results = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return sorted(results, key=lambda result: result.task_id)

    async def _run_one(
        self,
        request: ForkRequest,
        runner: SubAgentRunner,
    ) -> SubAgentResult:
        started = perf_counter()
        await self.monitor.emit(
            "subagent_started",
            f"Sub-agent {request.task_id} started.",
            tool="fork",
            payload={"subagent_id": request.task_id, "objective": request.objective},
        )
        try:
            async with self._semaphore:
                payload = await asyncio.wait_for(
                    runner(request),
                    timeout=request.timeout_seconds,
                )
            result = SubAgentResult(
                task_id=request.task_id,
                status="completed",
                result=payload.result,
                observations=payload.observations,
                warnings=payload.warnings,
                step_count=payload.step_count,
                elapsed_ms=int((perf_counter() - started) * 1000),
            )
        except asyncio.TimeoutError:
            result = SubAgentResult(
                task_id=request.task_id,
                status="timed_out",
                error=f"sub-agent timed out after {request.timeout_seconds}s",
                elapsed_ms=int((perf_counter() - started) * 1000),
            )
        except asyncio.CancelledError:
            await self.monitor.emit(
                "subagent_cancelled",
                f"Sub-agent {request.task_id} cancelled.",
                tool="fork",
                payload={"subagent_id": request.task_id},
            )
            raise
        except Exception as exc:
            result = SubAgentResult(
                task_id=request.task_id,
                status="failed",
                error=_safe_error(exc),
                elapsed_ms=int((perf_counter() - started) * 1000),
            )
        await self.monitor.emit(
            "subagent_finished",
            f"Sub-agent {request.task_id} finished with {result.status}.",
            tool="fork",
            payload=result.model_dump(),
        )
        return result
```

- [ ] **Step 4: Run executor tests**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_fork_executor_returns_stable_order_and_partial_failures tests/test_agent_loop.py::test_fork_executor_marks_timeout -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add app/agent/forking.py tests/test_agent_loop.py
git commit -m "feat: execute bounded homogeneous forks"
```

### Task 5: Run Child `CompetitionAgentLoop` Instances

**Files:**
- Modify: `app/agent/main_agent.py`
- Modify: `app/agent/forking.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `AgentScope`, `ForkExecutor`, `ScopedEventCollector`
- Produces: optional `scope: AgentScope | None = None` on `CompetitionAgentLoop.__init__()`
- Produces: fork observations in the parent trace

- [ ] **Step 1: Write a failing homogeneous-loop test**

Add this provider and test to `tests/test_agent_loop.py`:

```python
class ForkAwareLLMProvider(SequenceLLMProvider):
    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        system_prompt = str(messages[0].get("content", "")) if messages else ""
        if system_prompt.startswith("Extract shopping intent"):
            return await super().plan_next_action(messages)
        state = json.loads(messages[1]["content"])
        query = state["query"]
        completed = state["completed_actions"]
        if not completed:
            data = {"action": "plan", "arguments": {}, "thought": "Plan first."}
        elif query == "parent" and completed == ["plan"]:
            data = {
                "action": "fork",
                "thought": "Split independent platform checks.",
                "arguments": {
                    "tasks": [
                        {
                            "task_id": "amazon",
                            "objective": "child-amazon",
                            "allowed_tools": ["plan"],
                            "context_snapshot": {"platform": "Amazon"},
                            "merge_key": "products",
                        },
                        {
                            "task_id": "ebay",
                            "objective": "child-ebay",
                            "allowed_tools": ["plan"],
                            "context_snapshot": {"platform": "eBay"},
                            "merge_key": "products",
                        },
                    ]
                },
            }
        else:
            data = {"action": "finish", "message": "scope complete"}
        return ProviderResult(
            provider="fork_llm",
            provider_mode="fake",
            latency_ms=1,
            data=data,
        )


@pytest.mark.asyncio
async def test_main_loop_forks_same_loop_and_emits_one_root_result(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = ProviderRegistry(
        llm=ForkAwareLLMProvider([]),
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_homogeneous")
    loop = CompetitionAgentLoop(
        thread_id="thread_homogeneous",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("parent")

    starts = [event for event in monitor.events if event.type == "subagent_started"]
    results = [event for event in monitor.events if event.type == "task_result"]
    assert [event.payload["subagent_id"] for event in starts] == ["amazon", "ebay"]
    assert len(results) == 1
    assert (tmp_path / "subagents" / "amazon" / "summary.json").exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["action"] == "fork" for row in rows)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_agent_loop.py::test_main_loop_forks_same_loop_and_emits_one_root_result -q
```

Expected: FAIL because `CompetitionAgentLoop` does not execute `fork`.

- [ ] **Step 3: Add scope to the loop and Tool Registry**

Update imports and constructor fields in `app/agent/main_agent.py`:

```python
from app.agent.forking import AgentScope, ForkExecutor, ForkRequest, SubAgentPayload, SubAgentResult
from app.api.monitor import EventEmitter, ScopedEventCollector
```

Change `monitor` to `EventEmitter`, add the optional scope, and initialize it:

```python
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
```

In `run()`, construct the registry with the scope allowlist:

```python
        tools = ToolRegistry(ctx, allowed_tools=self.scope.allowed_tools)
```

Guard root-only start and result events:

```python
        if self.scope.emit_task_result:
            await self.monitor.emit(
                "task_started",
                "Competition Agent started.",
                payload={
                    "profile": self.settings.profile,
                    "provider_modes": self.settings.provider_modes(),
                },
            )
```

```python
        if self.scope.emit_task_result:
            await self.monitor.emit(
                "task_result",
                "Shopping summary generated.",
                payload={"summary": summary.model_dump()},
            )
```

- [ ] **Step 4: Execute `fork` before normal tool dispatch**

Insert this branch after terminal-action handling and before `tool_start`:

```python
            if action.name == "fork":
                results = await self._execute_fork(action)
                observation = {
                    "tool": "fork",
                    "subagents": [result.model_dump() for result in results],
                }
                ctx.observations.append(observation)
                steps.append(
                    AgentStep(
                        action=action,
                        observation_count=len(ctx.observations),
                        observations=[planner_observation, observation],
                    )
                )
                trace.append(
                    self._trace_row(
                        action,
                        [planner_observation, observation],
                        len(ctx.observations),
                    )
                )
                continue
```

Add this method to `CompetitionAgentLoop`:

```python
    async def _execute_fork(self, action: AgentAction) -> list[SubAgentResult]:
        requests = ForkRequest.parse_many(action.arguments, self.settings)
        if self.scope.depth >= self.settings.max_fork_depth:
            return [
                SubAgentResult(
                    task_id=request.task_id,
                    status="failed",
                    error=(
                        f"fork depth {self.scope.depth + 1} exceeds "
                        f"max_fork_depth={self.settings.max_fork_depth}"
                    ),
                )
                for request in requests
            ]
        executor = ForkExecutor(
            monitor=self.monitor,
            max_parallel=self.settings.max_parallel_subagents,
        )

        async def run_child(request: ForkRequest) -> SubAgentPayload:
            child_dir = self.session_dir / "subagents" / request.task_id
            child_monitor = ScopedEventCollector(
                self.monitor.parent if isinstance(self.monitor, ScopedEventCollector) else self.monitor,
                {
                    "subagent_id": request.task_id,
                    "fork_depth": self.scope.depth + 1,
                },
            )
            child = CompetitionAgentLoop(
                thread_id=self.thread_id,
                session_dir=child_dir,
                settings=self.settings,
                providers=self.providers,
                monitor=child_monitor,
                max_steps=request.max_steps,
                scope=AgentScope(
                    depth=self.scope.depth + 1,
                    task_id=request.task_id,
                    allowed_tools=frozenset(request.allowed_tools),
                    emit_task_result=False,
                    context_snapshot=request.context_snapshot,
                ),
            )
            summary = await child.run(request.objective)
            return SubAgentPayload(
                result={
                    "merge_key": request.merge_key,
                    "summary": summary.model_dump(),
                },
                observations=child.last_observations,
                warnings=summary.warnings,
                step_count=child.last_step_count,
            )

        return await executor.execute(requests, run_child)
```

Update the planner system prompt so `fork` is offered only when allowed:

```python
        fork_instruction = (
            "Allowed orchestration action: fork. "
            if self.scope.depth < self.settings.max_fork_depth
            else "Fork is not allowed at this depth. "
        )
```

Concatenate `fork_instruction` after the allowed tool actions in the existing system message. Include `fork_depth` and the scope's read-only `context_snapshot` in the user JSON:

```python
                            "fork_depth": self.scope.depth,
                            "context_snapshot": dict(self.scope.context_snapshot),
```

Immediately before building the summary at the end of `run()`, expose the completed child-run telemetry used by `SubAgentPayload`:

```python
        self.last_observations = list(ctx.observations)
        self.last_step_count = len(steps)
```

- [ ] **Step 5: Run the homogeneous-loop test and existing agent tests**

Run:

```bash
uv run pytest tests/test_agent_loop.py -q
```

Expected: all agent-loop tests PASS, including one root `task_result` and child output directories.

- [ ] **Step 6: Commit**

```bash
git add app/agent/main_agent.py app/agent/forking.py tests/test_agent_loop.py
git commit -m "feat: fork homogeneous competition agent loops"
```

### Task 6: Remove The Historical Prototype And Verify The Phase

**Files:**
- Delete: `app/agent/dispatch_tool.py`
- Modify: `README.md`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Removes: `dispatch_platform_search()` and `_search_platform()`
- Preserves: event names `subagent_started` and `subagent_finished` through `ForkExecutor`

- [ ] **Step 1: Prove no runtime import depends on the prototype**

Run:

```bash
rg -n "dispatch_platform_search|app\.agent\.dispatch_tool" app tests examples
```

Expected: only definitions in `app/agent/dispatch_tool.py`; no imports or calls.

- [ ] **Step 2: Delete the prototype**

Delete `app/agent/dispatch_tool.py`. Add this paragraph under the backend feature list in `README.md`:

```markdown
Homogeneous sub-agent work uses bounded forks of `CompetitionAgentLoop`. Each child has
isolated tool state, an allowlist, step/time budgets, scoped events, and structured merge
results; the removed `dispatch_tool.py` function-level mock is no longer the sub-agent path.
```

- [ ] **Step 3: Run focused verification**

Run:

```bash
uv run pytest tests/test_agent_loop.py tests/test_tools.py tests/test_config.py -q
```

Expected: all focused tests PASS.

- [ ] **Step 4: Run full regression verification**

Run:

```bash
uv run pytest -q
```

Expected: the full backend suite passes with zero failures.

Run:

```bash
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
```

Expected: exit code `0`; the summary discloses placeholder evidence.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_agent_loop.py
git add -u app/agent/dispatch_tool.py
git commit -m "refactor: remove mock subagent dispatcher"
```

## Phase Acceptance Checklist

- [ ] A root loop can select `fork` through the normal LLM Action path.
- [ ] Every child is a `CompetitionAgentLoop`, not a function-level task with a sub-agent label.
- [ ] Fork depth, parallelism, child steps, and child timeout are configured and tested.
- [ ] Child Tool Registries and observations are isolated.
- [ ] Results are sorted by `task_id` before parent observation construction.
- [ ] Partial failures are preserved without discarding successful children.
- [ ] Parent cancellation cancels unfinished children.
- [ ] Only the root emits `task_result`.
- [ ] `dispatch_tool.py` is removed after import and behavior checks pass.
- [ ] The complete backend suite and submission smoke pass.
