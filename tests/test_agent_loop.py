import asyncio
from dataclasses import replace
import json

import pytest

from app.agent.actions import AgentAction
from app.agent.forking import (
    AgentScope,
    ForkExecutor,
    ForkRequest,
    SubAgentPayload,
    thaw_context_snapshot,
)
from app.agent.main_agent import CompetitionAgentLoop
from app.api.monitor import EventCollector, ScopedEventCollector
from app.config import OmniMatchSettings
from app.providers.base import ProviderResult
from app.providers.registry import ProviderRegistry


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


class SequenceLLMProvider:
    def __init__(self, actions: list[dict]) -> None:
        self.actions = actions
        self.calls: list[list[dict]] = []
        self.action_calls: list[list[dict]] = []

    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        self.calls.append(messages)
        system_prompt = str(messages[0].get("content", "")) if messages else ""
        if system_prompt.startswith("Extract shopping intent"):
            return ProviderResult(
                provider="sequence_llm",
                provider_mode="fake",
                latency_ms=1,
                data={
                    "action": "plan_query",
                    "arguments": {
                        "category": "旅行三件套",
                        "budget": 300,
                        "preferences": [],
                        "negative_constraints": ["塑料"],
                        "destination": None,
                    },
                },
                response_summary="sequence intent",
            )
        self.action_calls.append(messages)
        index = min(len(self.action_calls) - 1, len(self.actions) - 1)
        return ProviderResult(
            provider="sequence_llm",
            provider_mode="fake",
            latency_ms=1,
            data=self.actions[index],
            response_summary=f"sequence action {index}",
        )


class ForkAwareLLMProvider(SequenceLLMProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.planner_states: dict[str, list[dict]] = {}
        self.system_prompts: dict[str, list[str]] = {}

    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        system_prompt = str(messages[0].get("content", "")) if messages else ""
        if system_prompt.startswith("Extract shopping intent"):
            return await super().plan_next_action(messages)

        self.calls.append(messages)
        self.action_calls.append(messages)
        state = json.loads(messages[1]["content"])
        query = state["query"]
        completed = state["completed_actions"]
        self.planner_states.setdefault(query, []).append(state)
        self.system_prompts.setdefault(query, []).append(system_prompt)
        if not completed:
            data = {"action": "plan", "arguments": {}, "thought": "Plan first."}
        elif query == "parent" and completed == ["plan"]:
            data = {
                "action": "fork",
                "thought": "Split independent platform checks.",
                "arguments": {
                    "tasks": [
                        {
                            "task_id": "ebay",
                            "objective": "child-ebay",
                            "allowed_tools": ["plan"],
                            "context_snapshot": {
                                "platform": "eBay",
                                "filters": {"regions": ["US"]},
                            },
                            "merge_key": "products",
                        },
                        {
                            "task_id": "amazon",
                            "objective": "child-amazon",
                            "allowed_tools": ["plan"],
                            "context_snapshot": {
                                "platform": "Amazon",
                                "filters": {"regions": ["US"]},
                            },
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


class SecretFailingChildLLMProvider(ForkAwareLLMProvider):
    def __init__(self, error_text: str) -> None:
        super().__init__()
        self.error_text = error_text

    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        system_prompt = str(messages[0].get("content", "")) if messages else ""
        if not system_prompt.startswith("Extract shopping intent"):
            state = json.loads(messages[1]["content"])
            if state["query"] == "child-amazon":
                raise RuntimeError(self.error_text)
        return await super().plan_next_action(messages)


class BudgetForkLLMProvider(SequenceLLMProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.planner_states: dict[str, list[dict]] = {}

    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        system_prompt = str(messages[0].get("content", "")) if messages else ""
        if system_prompt.startswith("Extract shopping intent"):
            return await super().plan_next_action(messages)

        self.calls.append(messages)
        self.action_calls.append(messages)
        state = json.loads(messages[1]["content"])
        query = state["query"]
        completed = state["completed_actions"]
        self.planner_states.setdefault(query, []).append(state)
        if query == "parent" and not completed:
            data = {"action": "plan", "arguments": {}, "thought": "Plan first."}
        elif query == "parent" and completed == ["plan"]:
            data = {
                "action": "fork",
                "arguments": {
                    "tasks": [
                        {
                            "task_id": "budget_child",
                            "objective": "budget-child",
                            "allowed_tools": ["plan"],
                            "context_snapshot": {},
                            "max_steps": 2,
                            "merge_key": "products",
                        }
                    ]
                },
            }
        elif query == "parent":
            data = {"action": "finish", "message": "scope complete"}
        else:
            data = {"action": "plan", "arguments": {}, "thought": "Keep planning."}
        return ProviderResult(
            provider="budget_fork_llm",
            provider_mode="fake",
            latency_ms=1,
            data=data,
        )


class ImmediateForkLLMProvider(SequenceLLMProvider):
    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        system_prompt = str(messages[0].get("content", "")) if messages else ""
        if system_prompt.startswith("Extract shopping intent"):
            return await super().plan_next_action(messages)

        self.calls.append(messages)
        self.action_calls.append(messages)
        state = json.loads(messages[1]["content"])
        if state["query"] == "parent" and not state["completed_actions"]:
            data = {
                "action": "fork",
                "arguments": {
                    "tasks": [
                        {
                            "task_id": "first_step_child",
                            "objective": "child",
                            "allowed_tools": ["plan"],
                            "context_snapshot": {},
                            "merge_key": "products",
                        }
                    ]
                },
            }
        else:
            data = {"action": "finish", "message": "scope complete"}
        return ProviderResult(
            provider="immediate_fork_llm",
            provider_mode="fake",
            latency_ms=1,
            data=data,
        )


class TerminalFailForkLLMProvider(ForkAwareLLMProvider):
    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        system_prompt = str(messages[0].get("content", "")) if messages else ""
        if not system_prompt.startswith("Extract shopping intent"):
            state = json.loads(messages[1]["content"])
            if state["query"] == "child-amazon":
                return ProviderResult(
                    provider="terminal_fail_fork_llm",
                    provider_mode="fake",
                    latency_ms=1,
                    data={
                        "action": "fail",
                        "message": "Child could not satisfy objective.",
                    },
                )
        return await super().plan_next_action(messages)


class SecretTerminalFailForkLLMProvider(ForkAwareLLMProvider):
    sentinel = "child-leak-sentinel"

    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        system_prompt = str(messages[0].get("content", "")) if messages else ""
        if not system_prompt.startswith("Extract shopping intent"):
            state = json.loads(messages[1]["content"])
            if state["query"] == "child-amazon":
                return ProviderResult(
                    provider="secret_terminal_fail_fork_llm",
                    provider_mode="fake",
                    latency_ms=1,
                    data={
                        "action": "fail",
                        "thought": f"Child saw token={self.sentinel}",
                        "message": f"Child failed with password={self.sentinel}",
                    },
                )
        return await super().plan_next_action(messages)


class NestedForkLLMProvider(SequenceLLMProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.planner_states: dict[str, list[dict]] = {}
        self.system_prompts: dict[str, list[str]] = {}

    async def plan_next_action(self, messages: list[dict]) -> ProviderResult[dict]:
        system_prompt = str(messages[0].get("content", "")) if messages else ""
        if system_prompt.startswith("Extract shopping intent"):
            return await super().plan_next_action(messages)

        self.calls.append(messages)
        self.action_calls.append(messages)
        state = json.loads(messages[1]["content"])
        query = state["query"]
        completed = state["completed_actions"]
        self.planner_states.setdefault(query, []).append(state)
        self.system_prompts.setdefault(query, []).append(system_prompt)
        if not completed:
            data = {"action": "plan", "arguments": {}, "thought": "Plan first."}
        elif query == "parent" and completed == ["plan"]:
            data = {
                "action": "fork",
                "arguments": {
                    "tasks": [
                        {
                            "task_id": "child",
                            "objective": "child",
                            "allowed_tools": ["plan"],
                            "context_snapshot": {"level": 1},
                            "merge_key": "children",
                        }
                    ]
                },
            }
        elif query == "child" and completed == ["plan"]:
            data = {
                "action": "fork",
                "arguments": {
                    "tasks": [
                        {
                            "task_id": "grandchild_b",
                            "objective": "grandchild-b",
                            "allowed_tools": ["plan"],
                            "context_snapshot": {"level": 2, "slot": "b"},
                            "merge_key": "grandchildren",
                        },
                        {
                            "task_id": "grandchild_a",
                            "objective": "grandchild-a",
                            "allowed_tools": ["plan"],
                            "context_snapshot": {"level": 2, "slot": "a"},
                            "merge_key": "grandchildren",
                        },
                    ]
                },
            }
        else:
            data = {"action": "finish", "message": "scope complete"}
        return ProviderResult(
            provider="nested_fork_llm",
            provider_mode="fake",
            latency_ms=1,
            data=data,
        )


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


@pytest.mark.parametrize(
    ("field", "value", "limit"),
    [("max_steps", 5, "subagent_max_steps"), ("timeout_seconds", 30.1, "subagent_timeout_seconds")],
)
def test_fork_request_rejects_budget_above_settings(field, value, limit):
    with pytest.raises(ValueError, match=limit):
        ForkRequest.parse_many(
            {
                "tasks": [
                    {
                        "task_id": "amazon",
                        "objective": "Search Amazon",
                        "allowed_tools": ["plan"],
                        "context_snapshot": {},
                        field: value,
                        "merge_key": "products",
                    }
                ]
            },
            submission_settings(),
        )


def test_fork_request_allows_lower_budgets():
    request = ForkRequest.parse_many(
        {
            "tasks": [
                {
                    "task_id": "amazon",
                    "objective": "Search Amazon",
                    "allowed_tools": ["plan"],
                    "context_snapshot": {},
                    "max_steps": 2,
                    "timeout_seconds": 10.0,
                    "merge_key": "products",
                }
            ]
        },
        submission_settings(),
    )[0]

    assert request.max_steps == 2
    assert request.timeout_seconds == 10.0


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
async def test_fork_executor_preserves_typed_failed_payload_data():
    executor = ForkExecutor(
        monitor=EventCollector(thread_id="thread_typed_failure"),
        max_parallel=1,
    )
    request = ForkRequest(
        task_id="failed_child",
        objective="fail normally",
        allowed_tools=["plan"],
        context_snapshot={},
        max_steps=1,
        timeout_seconds=1,
        merge_key="products",
    )

    async def runner(_: ForkRequest) -> SubAgentPayload:
        return SubAgentPayload(
            status="failed",
            error="Child could not satisfy objective.",
            result={"summary": {"status_note": "Child could not satisfy objective."}},
            observations=[{"tool": "AgentPlanner"}],
            step_count=1,
        )

    result = (await executor.execute([request], runner))[0]

    assert result.status == "failed"
    assert result.error == "Child could not satisfy objective."
    assert result.result == {"summary": {"status_note": "Child could not satisfy objective."}}
    assert result.observations == [{"tool": "AgentPlanner"}]
    assert result.step_count == 1


def test_subagent_payload_rejects_failed_without_error():
    with pytest.raises(ValueError, match="failed sub-agent payload requires an error"):
        SubAgentPayload(status="failed", result={})


def test_subagent_payload_rejects_completed_with_error():
    with pytest.raises(ValueError, match="completed sub-agent payload cannot include an error"):
        SubAgentPayload(status="completed", error="unexpected", result={})


def test_subagent_payload_defaults_to_completed_without_error():
    payload = SubAgentPayload(result={})

    assert payload.status == "completed"
    assert payload.error is None


@pytest.mark.asyncio
async def test_fork_executor_marks_timeout():
    monitor = EventCollector(thread_id="thread_timeout")
    executor = ForkExecutor(monitor=monitor, max_parallel=2)
    requests = [
        ForkRequest(
            task_id="slow",
            objective="slow",
            allowed_tools=["plan"],
            context_snapshot={},
            max_steps=1,
            timeout_seconds=0.01,
            merge_key="products",
        ),
        ForkRequest(
            task_id="fast",
            objective="fast",
            allowed_tools=["plan"],
            context_snapshot={},
            max_steps=1,
            timeout_seconds=1,
            merge_key="products",
        ),
    ]

    async def runner(request: ForkRequest) -> SubAgentPayload:
        if request.task_id == "slow":
            await asyncio.sleep(1)
        return SubAgentPayload(result={"task_id": request.task_id})

    results = await executor.execute(requests, runner)

    assert [result.status for result in results] == ["completed", "timed_out"]
    assert "timed out" in (results[1].error or "")
    assert {
        (event.type, event.payload["task_id"], event.payload["status"])
        for event in monitor.events
        if event.type == "subagent_finished"
    } == {("subagent_finished", "fast", "completed")}


@pytest.mark.asyncio
async def test_fork_executor_timeout_includes_semaphore_queue_wait():
    queued_started = asyncio.Event()

    async def on_event(event) -> None:
        if event.type == "subagent_started" and event.payload["subagent_id"] == "queued":
            queued_started.set()

    monitor = EventCollector(thread_id="thread_queued_timeout", sink=on_event)
    executor = ForkExecutor(monitor=monitor, max_parallel=1)
    requests = [
        ForkRequest(
            task_id="blocker",
            objective="blocker",
            allowed_tools=["plan"],
            context_snapshot={},
            max_steps=1,
            timeout_seconds=1,
            merge_key="products",
        ),
        ForkRequest(
            task_id="queued",
            objective="queued",
            allowed_tools=["plan"],
            context_snapshot={},
            max_steps=1,
            timeout_seconds=0.01,
            merge_key="products",
        ),
    ]
    blocker_started = asyncio.Event()
    release_blocker = asyncio.Event()
    runner_started: set[str] = set()

    async def runner(request: ForkRequest) -> SubAgentPayload:
        runner_started.add(request.task_id)
        if request.task_id == "blocker":
            blocker_started.set()
            await release_blocker.wait()
        return SubAgentPayload(result={"task_id": request.task_id})

    execution = asyncio.create_task(executor.execute(requests, runner))
    await asyncio.wait_for(blocker_started.wait(), timeout=1)
    await asyncio.wait_for(queued_started.wait(), timeout=1)
    try:
        # The queued child has a 10ms whole-lifecycle budget. This tolerance is
        # deliberately larger because this test is exercising real timeout behavior.
        await asyncio.sleep(0.05)
    finally:
        release_blocker.set()
        results = await asyncio.wait_for(execution, timeout=1)

    assert runner_started == {"blocker"}
    assert [(result.task_id, result.status) for result in results] == [
        ("blocker", "completed"),
        ("queued", "timed_out"),
    ]


@pytest.mark.asyncio
async def test_fork_executor_timeout_includes_slow_start_emission():
    start_emit_entered = asyncio.Event()
    start_emit_stopped = asyncio.Event()
    block_start_emit = asyncio.Event()
    runner_started = asyncio.Event()
    emit_attempts: list[str] = []

    async def on_event(event) -> None:
        emit_attempts.append(event.type)
        if event.type != "subagent_started":
            return
        start_emit_entered.set()
        try:
            await block_start_emit.wait()
        finally:
            start_emit_stopped.set()

    monitor = EventCollector(thread_id="thread_slow_start_emit", sink=on_event)
    executor = ForkExecutor(monitor=monitor, max_parallel=1)
    request = ForkRequest(
        task_id="slow_start",
        objective="never reaches runner",
        allowed_tools=["plan"],
        context_snapshot={},
        max_steps=1,
        timeout_seconds=0.02,
        merge_key="products",
    )

    async def runner(_: ForkRequest) -> SubAgentPayload:
        runner_started.set()
        return SubAgentPayload(result={})

    execution = asyncio.create_task(executor.execute([request], runner))
    await asyncio.wait_for(start_emit_entered.wait(), timeout=1)
    results = await asyncio.wait_for(execution, timeout=0.25)
    await asyncio.wait_for(start_emit_stopped.wait(), timeout=1)

    assert [(result.task_id, result.status) for result in results] == [("slow_start", "timed_out")]
    assert runner_started.is_set() is False
    assert emit_attempts == ["subagent_started"]
    assert not any(event.type == "subagent_started" for event in monitor.events)


@pytest.mark.asyncio
async def test_fork_executor_timeout_includes_slow_finish_emission():
    finish_emit_entered = asyncio.Event()
    finish_emit_stopped = asyncio.Event()
    block_finish_emit = asyncio.Event()
    runner_stopped = asyncio.Event()
    emit_attempts: list[str] = []

    async def on_event(event) -> None:
        emit_attempts.append(event.type)
        if event.type != "subagent_finished":
            return
        finish_emit_entered.set()
        try:
            await block_finish_emit.wait()
        finally:
            finish_emit_stopped.set()

    monitor = EventCollector(thread_id="thread_slow_finish_emit", sink=on_event)
    executor = ForkExecutor(monitor=monitor, max_parallel=1)
    request = ForkRequest(
        task_id="slow_finish",
        objective="runner completes",
        allowed_tools=["plan"],
        context_snapshot={},
        max_steps=1,
        timeout_seconds=0.02,
        merge_key="products",
    )

    async def runner(_: ForkRequest) -> SubAgentPayload:
        runner_stopped.set()
        return SubAgentPayload(result={})

    execution = asyncio.create_task(executor.execute([request], runner))
    await asyncio.wait_for(finish_emit_entered.wait(), timeout=1)
    results = await asyncio.wait_for(execution, timeout=0.25)
    await asyncio.wait_for(finish_emit_stopped.wait(), timeout=1)

    assert [(result.task_id, result.status) for result in results] == [("slow_finish", "completed")]
    assert runner_stopped.is_set()
    assert emit_attempts == ["subagent_started", "subagent_finished"]
    assert not any(
        event.type == "subagent_finished" and event.payload["status"] == "completed"
        for event in monitor.events
    )


@pytest.mark.asyncio
async def test_fork_executor_partial_finish_delivery_preserves_computed_status():
    canonical_events = []
    delivered_payloads: list[dict] = []
    finish_delivery_started = asyncio.Event()
    finish_delivery_stopped = asyncio.Event()
    block_remaining_delivery = asyncio.Event()

    async def partially_delivering_sink(event) -> None:
        if event.type != "subagent_finished":
            return
        delivered_payloads.append(dict(event.payload))
        finish_delivery_started.set()
        try:
            await block_remaining_delivery.wait()
        finally:
            finish_delivery_stopped.set()

    monitor = EventCollector(
        thread_id="thread_partial_finish_delivery",
        sink=partially_delivering_sink,
        events=canonical_events,
    )
    executor = ForkExecutor(monitor=monitor, max_parallel=1)
    request = ForkRequest(
        task_id="partial_finish",
        objective="runner completes before notification deadline",
        allowed_tools=["plan"],
        context_snapshot={},
        max_steps=1,
        timeout_seconds=0.02,
        merge_key="products",
    )

    async def runner(_: ForkRequest) -> SubAgentPayload:
        return SubAgentPayload(result={"computed": True}, step_count=1)

    execution = asyncio.create_task(executor.execute([request], runner))
    await asyncio.wait_for(finish_delivery_started.wait(), timeout=1)
    result = (await asyncio.wait_for(execution, timeout=0.25))[0]
    await asyncio.wait_for(finish_delivery_stopped.wait(), timeout=1)

    assert result.status == "completed"
    assert result.result == {"computed": True}
    assert delivered_payloads[0]["status"] == result.status
    assert [event.type for event in canonical_events] == ["subagent_started"]


@pytest.mark.asyncio
async def test_fork_executor_emitter_failure_cancels_and_awaits_siblings():
    sibling_runner_started = asyncio.Event()
    sibling_runner_stopped = asyncio.Event()
    release_sibling = asyncio.Event()
    terminal_events = {
        "sibling": asyncio.Event(),
        "queued": asyncio.Event(),
    }

    async def on_event(event) -> None:
        if event.type == "subagent_started" and event.payload["subagent_id"] == "broken":
            await sibling_runner_started.wait()
            raise RuntimeError("event sink exploded")
        if event.type not in {"subagent_finished", "subagent_cancelled"}:
            return
        task_id = event.payload.get("task_id") or event.payload.get("subagent_id")
        if task_id in terminal_events:
            terminal_events[task_id].set()

    monitor = EventCollector(thread_id="thread_emitter_failure", sink=on_event)
    executor = ForkExecutor(monitor=monitor, max_parallel=1)
    requests = [
        ForkRequest(
            task_id=task_id,
            objective="wait",
            allowed_tools=["plan"],
            context_snapshot={},
            max_steps=1,
            timeout_seconds=1,
            merge_key="products",
        )
        for task_id in ("broken", "sibling", "queued")
    ]
    runner_started: set[str] = set()

    async def runner(request: ForkRequest) -> SubAgentPayload:
        runner_started.add(request.task_id)
        if request.task_id == "sibling":
            sibling_runner_started.set()
            try:
                await release_sibling.wait()
            finally:
                sibling_runner_stopped.set()
        return SubAgentPayload(result={"task_id": request.task_id})

    try:
        with pytest.raises(RuntimeError, match="event sink exploded"):
            await asyncio.wait_for(executor.execute(requests, runner), timeout=1)
        assert sibling_runner_stopped.is_set()
        assert runner_started == {"sibling"}
    finally:
        release_sibling.set()
        await asyncio.wait_for(terminal_events["sibling"].wait(), timeout=1)
        await asyncio.wait_for(terminal_events["queued"].wait(), timeout=1)


@pytest.mark.asyncio
async def test_fork_executor_parent_cancellation_cancels_queued_and_running_children():
    monitor = EventCollector(thread_id="thread_cancel")
    executor = ForkExecutor(monitor=monitor, max_parallel=1)
    requests = [
        ForkRequest(
            task_id=task_id,
            objective="wait",
            allowed_tools=["plan"],
            context_snapshot={},
            max_steps=1,
            timeout_seconds=10,
            merge_key="products",
        )
        for task_id in ("running", "queued_a", "queued_b")
    ]
    running_started = asyncio.Event()
    running_stopped = asyncio.Event()
    runner_started: set[str] = set()

    async def runner(request: ForkRequest) -> SubAgentPayload:
        runner_started.add(request.task_id)
        if request.task_id == "running":
            running_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            if request.task_id == "running":
                running_stopped.set()
        return SubAgentPayload(result={})

    execution = asyncio.create_task(executor.execute(requests, runner))
    await asyncio.wait_for(running_started.wait(), timeout=1)

    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution
    await asyncio.wait_for(running_stopped.wait(), timeout=1)
    assert runner_started == {"running"}
    assert {
        event.payload["subagent_id"]
        for event in monitor.events
        if event.type == "subagent_cancelled"
    } == {"running", "queued_a", "queued_b"}


@pytest.mark.asyncio
async def test_fork_executor_never_exceeds_max_parallel_runners():
    executor = ForkExecutor(monitor=EventCollector(thread_id="thread_parallel"), max_parallel=2)
    requests = [
        ForkRequest(
            task_id=task_id,
            objective="wait",
            allowed_tools=["plan"],
            context_snapshot={},
            max_steps=1,
            timeout_seconds=1,
            merge_key="products",
        )
        for task_id in ("a", "b", "c")
    ]
    two_runners_active = asyncio.Event()
    release_runners = asyncio.Event()
    active_runners = 0
    peak_runners = 0
    runner_started: set[str] = set()

    async def runner(request: ForkRequest) -> SubAgentPayload:
        nonlocal active_runners, peak_runners
        active_runners += 1
        peak_runners = max(peak_runners, active_runners)
        runner_started.add(request.task_id)
        if active_runners == 2:
            two_runners_active.set()
        try:
            await release_runners.wait()
        finally:
            active_runners -= 1
        return SubAgentPayload(result={"task_id": request.task_id})

    execution = asyncio.create_task(executor.execute(requests, runner))
    await asyncio.wait_for(two_runners_active.wait(), timeout=1)

    assert active_runners == 2
    assert peak_runners == 2
    assert len(runner_started) == 2

    release_runners.set()
    await execution

    assert peak_runners == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_text",
    [
        "bEaReR SECRET+/=TOKEN",
        "api key: SECRET+/=TOKEN",
        "api_key = SECRET+/=TOKEN",
        "api-key=SECRET+/=TOKEN",
        "OPENAI_API_KEY=SECRET+/=TOKEN",
        "serpapi_api_key : SECRET+/=TOKEN",
        "Aws_Access_Token = SECRET+/=TOKEN",
        "password : SECRET+/=TOKEN",
        "AUTHORIZATION=SECRET+/=TOKEN",
        "token = SECRET+/=TOKEN",
    ],
)
async def test_fork_executor_redacts_secret_forms_from_results_and_events(error_text):
    secret = "SECRET+/=TOKEN"
    monitor = EventCollector(thread_id="thread_secret")
    executor = ForkExecutor(monitor=monitor, max_parallel=1)
    request = ForkRequest(
        task_id="secret",
        objective="fail safely",
        allowed_tools=["plan"],
        context_snapshot={},
        max_steps=1,
        timeout_seconds=1,
        merge_key="products",
    )

    async def runner(_: ForkRequest) -> SubAgentPayload:
        raise RuntimeError(f"provider failed: {error_text}")

    result = (await executor.execute([request], runner))[0]
    finished_payload = next(
        event.payload for event in monitor.events if event.type == "subagent_finished"
    )

    assert "[REDACTED]" in (result.error or "")
    assert secret not in (result.error or "")
    assert secret not in json.dumps(finished_payload)


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


@pytest.mark.asyncio
async def test_event_collector_rolls_back_exact_event_identity_when_sink_fails():
    replay_events = []
    emitted_event = None
    equal_event = None

    async def failing_sink(event) -> None:
        nonlocal emitted_event, equal_event
        emitted_event = event
        equal_event = event.model_copy(deep=True)
        monitor.events.insert(0, equal_event)
        raise RuntimeError("delivery failed")

    monitor = EventCollector(
        thread_id="thread_transactional_event",
        sink=failing_sink,
        events=replay_events,
    )

    with pytest.raises(RuntimeError, match="delivery failed"):
        await monitor.emit("tool_start", "search", tool="item_search")

    assert monitor.events is replay_events
    assert equal_event == emitted_event
    assert equal_event is not emitted_event
    assert len(monitor.events) == 1
    assert monitor.events[0] is equal_event


@pytest.mark.asyncio
async def test_scoped_event_collector_scope_keys_override_event_payload():
    parent = EventCollector(thread_id="thread_scope_conflict")
    child = ScopedEventCollector(
        parent,
        {"subagent_id": "amazon", "fork_depth": 1},
    )

    await child.emit(
        "tool_start",
        "search",
        tool="item_search",
        payload={"subagent_id": "parent", "fork_depth": 0},
    )

    assert parent.events[-1].payload == {
        "subagent_id": "amazon",
        "fork_depth": 1,
    }


def test_agent_scope_context_snapshot_is_read_only():
    scope = AgentScope(context_snapshot={"platform": "Amazon"})

    with pytest.raises(TypeError):
        scope.context_snapshot["platform"] = "eBay"


def test_agent_scope_context_snapshot_is_recursively_read_only():
    scope = AgentScope(
        context_snapshot={"platforms": ["Amazon"], "filters": {"price": {"max": 300}}}
    )

    with pytest.raises(TypeError):
        scope.context_snapshot["filters"]["price"]["max"] = 500
    with pytest.raises(AttributeError):
        scope.context_snapshot["platforms"].append("eBay")


def test_thaw_context_snapshot_returns_mutable_json_safe_copy():
    scope = AgentScope(context_snapshot={"platforms": ["Amazon"], "tags": ["travel"]})

    snapshot = thaw_context_snapshot(scope.context_snapshot)
    snapshot["platforms"].append("eBay")

    assert snapshot == {"platforms": ["Amazon", "eBay"], "tags": ["travel"]}
    assert json.dumps(snapshot)


@pytest.mark.parametrize("invalid_value", [bytearray(b"unsafe"), {"travel"}])
def test_agent_scope_rejects_non_json_snapshot_values(invalid_value):
    with pytest.raises(ValueError, match="context_snapshot.*not JSON-compatible"):
        AgentScope(context_snapshot={"nested": {"value": invalid_value}})


def test_agent_scope_rejects_non_string_context_keys():
    with pytest.raises(ValueError, match="context_snapshot.*keys must be strings"):
        AgentScope(context_snapshot={1: "Amazon"})


def test_fork_request_rejects_non_json_snapshot_value():
    with pytest.raises(ValueError, match="context_snapshot.*not JSON-compatible"):
        ForkRequest.parse_many(
            {
                "tasks": [
                    {
                        "task_id": "amazon",
                        "objective": "Search Amazon",
                        "allowed_tools": ["plan"],
                        "context_snapshot": {"nested": {"value": bytearray(b"unsafe")}},
                        "merge_key": "products",
                    }
                ]
            },
            submission_settings(),
        )


@pytest.mark.asyncio
async def test_root_loop_executes_fork_on_first_step(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = ProviderRegistry(
        llm=ImmediateForkLLMProvider([]),
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_first_step_fork")
    loop = CompetitionAgentLoop(
        thread_id="thread_first_step_fork",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("parent")

    rows = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["action"] for row in rows] == ["fork", "finish"]
    assert [
        event.payload["subagent_id"] for event in monitor.events if event.type == "subagent_started"
    ] == ["first_step_child"]
    assert not [event for event in monitor.events if event.type == "tool_start"]


@pytest.mark.asyncio
async def test_child_terminal_fail_is_scoped_and_returns_failed_result(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = ProviderRegistry(
        llm=TerminalFailForkLLMProvider(),
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_child_terminal_fail")
    loop = CompetitionAgentLoop(
        thread_id="thread_child_terminal_fail",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("parent")

    fork_observation = next(
        observation for observation in loop.last_observations if observation["tool"] == "fork"
    )
    failed_result = next(
        result for result in fork_observation["subagents"] if result["task_id"] == "amazon"
    )
    child_errors = [event for event in monitor.events if event.type == "subagent_error"]
    assert failed_result["status"] == "failed"
    assert failed_result["error"] == "Child could not satisfy objective."
    assert failed_result["result"]["summary"]["status_note"] == failed_result["error"]
    assert failed_result["observations"]
    assert failed_result["step_count"] == 1
    assert [event.message for event in child_errors] == [failed_result["error"]]
    assert child_errors[0].payload["subagent_id"] == "amazon"
    assert child_errors[0].payload["fork_depth"] == 1
    assert not [event for event in monitor.events if event.type == "task_error"]


@pytest.mark.asyncio
async def test_child_terminal_fail_is_sanitized_before_every_event_and_artifact(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    llm = SecretTerminalFailForkLLMProvider()
    providers = ProviderRegistry(
        llm=llm,
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_child_terminal_redaction")
    loop = CompetitionAgentLoop(
        thread_id="thread_child_terminal_redaction",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("parent")

    child_events = [
        event
        for event in monitor.events
        if event.payload.get("subagent_id") == "amazon"
        or (event.payload.get("task_id") == "amazon" and event.payload.get("fork_depth") == 1)
    ]
    child_trace = (tmp_path / "subagents" / "amazon" / "trace.jsonl").read_text(encoding="utf-8")
    child_summary = (tmp_path / "subagents" / "amazon" / "summary.json").read_text(encoding="utf-8")
    finished_payload = next(
        event.payload for event in child_events if event.type == "subagent_finished"
    )
    parent_fork_observation = next(
        observation for observation in loop.last_observations if observation["tool"] == "fork"
    )

    assert child_events
    assert any(event.type == "thought" and "[REDACTED]" in event.message for event in child_events)
    for event in child_events:
        assert llm.sentinel not in event.model_dump_json()
    for artifact in (
        child_trace,
        child_summary,
        json.dumps(finished_payload),
        json.dumps(parent_fork_observation),
    ):
        assert llm.sentinel not in artifact


@pytest.mark.asyncio
async def test_main_loop_forks_same_loop_and_isolates_child_state(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    llm = ForkAwareLLMProvider()
    providers = ProviderRegistry(
        llm=llm,
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
    task_starts = [event for event in monitor.events if event.type == "task_started"]
    task_results = [event for event in monitor.events if event.type == "task_result"]
    assert [event.payload["subagent_id"] for event in starts] == ["ebay", "amazon"]
    assert len(task_starts) == 1
    assert len(task_results) == 1

    root_rows = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    root_fork = next(
        observation
        for row in root_rows
        for observation in row["observations"]
        if observation["tool"] == "fork"
    )
    assert [result["task_id"] for result in root_fork["subagents"]] == ["amazon", "ebay"]
    assert [result["status"] for result in root_fork["subagents"]] == [
        "completed",
        "completed",
    ]

    assert llm.planner_states["child-amazon"][0]["fork_depth"] == 1
    assert llm.planner_states["child-amazon"][0]["context_snapshot"] == {
        "platform": "Amazon",
        "filters": {"regions": ["US"]},
    }
    assert "Allowed orchestration action: fork." in llm.system_prompts["parent"][0]
    assert "Fork is not allowed at this depth." in llm.system_prompts["child-amazon"][0]

    parent_fork = next(
        observation for observation in loop.last_observations if observation["tool"] == "fork"
    )
    assert [result["task_id"] for result in parent_fork["subagents"]] == ["amazon", "ebay"]
    for result in parent_fork["subagents"]:
        assert result["observations"] is not loop.last_observations
        assert not any(observation["tool"] == "fork" for observation in result["observations"])

    for task_id in ("amazon", "ebay"):
        child_dir = tmp_path / "subagents" / task_id
        assert (child_dir / "summary.json").exists()
        child_rows = [
            json.loads(line)
            for line in (child_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [row["action"] for row in child_rows] == ["plan", "finish"]
    assert [row["action"] for row in root_rows] == ["plan", "fork", "finish"]


@pytest.mark.asyncio
async def test_fork_integration_redacts_child_error_from_result_event_and_parent_trace(tmp_path):
    secrets = {
        "sk-openai-secret",
        "serp-secret",
        "aws-secret",
        "password-secret",
        "authorization-secret",
        "token-secret",
    }
    error_text = " ".join(
        [
            "OPENAI_API_KEY=sk-openai-secret",
            "SERPAPI_API_KEY:serp-secret",
            "AWS_ACCESS_TOKEN=aws-secret",
            "password=password-secret",
            "authorization:Bearer authorization-secret",
            "token=token-secret",
        ]
    )
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = ProviderRegistry(
        llm=SecretFailingChildLLMProvider(error_text),
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_secret_trace")
    loop = CompetitionAgentLoop(
        thread_id="thread_secret_trace",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("parent")

    fork_observation = next(
        observation for observation in loop.last_observations if observation["tool"] == "fork"
    )
    failed_result = next(
        result for result in fork_observation["subagents"] if result["task_id"] == "amazon"
    )
    finished_payload = next(
        event.payload
        for event in monitor.events
        if event.type == "subagent_finished" and event.payload["task_id"] == "amazon"
    )
    parent_trace = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")

    assert failed_result["status"] == "failed"
    assert "[REDACTED]" in failed_result["error"]
    for secret in secrets:
        assert secret not in failed_result["error"]
        assert secret not in json.dumps(finished_payload)
        assert secret not in parent_trace


@pytest.mark.asyncio
async def test_child_planner_prompt_discloses_only_scoped_tools(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    llm = ForkAwareLLMProvider()
    providers = ProviderRegistry(
        llm=llm,
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    loop = CompetitionAgentLoop(
        thread_id="thread_prompt_scope",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=EventCollector(thread_id="thread_prompt_scope"),
    )

    await loop.run("parent")

    root_prompt = llm.system_prompts["parent"][0]
    child_prompt = llm.system_prompts["child-amazon"][0]
    assert (
        "Allowed tool actions: plan, category_insight, item_search, shipping, rank, pick."
        in root_prompt
    )
    assert "Allowed tool actions: plan." in child_prompt
    for disallowed_tool in ("category_insight", "item_search", "shipping", "rank", "pick"):
        assert disallowed_tool not in child_prompt


@pytest.mark.asyncio
async def test_fork_integration_enforces_lower_child_step_budget_and_reports_exact_count(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    llm = BudgetForkLLMProvider()
    providers = ProviderRegistry(
        llm=llm,
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_child_budget")
    loop = CompetitionAgentLoop(
        thread_id="thread_child_budget",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("parent")

    fork_observation = next(
        observation for observation in loop.last_observations if observation["tool"] == "fork"
    )
    child_result = fork_observation["subagents"][0]
    child_trace = [
        json.loads(line)
        for line in (tmp_path / "subagents" / "budget_child" / "trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert settings.subagent_max_steps == 4
    assert len(llm.planner_states["budget-child"]) == 2
    assert [row["action"] for row in child_trace] == ["plan", "plan"]
    assert child_result["status"] == "failed"
    assert child_result["error"] == "Reached max_steps=2 before a finish action."
    assert child_result["step_count"] == 2
    assert child_result["result"]["summary"]["status_note"] == (
        "Reached max_steps=2 before a finish action."
    )
    child_errors = [event for event in monitor.events if event.type == "subagent_error"]
    assert len(child_errors) == 1
    assert child_errors[0].payload["subagent_id"] == "budget_child"
    assert child_errors[0].payload["fork_depth"] == 1
    assert not [event for event in monitor.events if event.type == "task_error"]


@pytest.mark.asyncio
async def test_child_fork_at_depth_limit_returns_sorted_failures_without_grandchildren(tmp_path):
    settings = replace(submission_settings(), max_fork_depth=1)
    base = ProviderRegistry.from_settings(settings)
    llm = NestedForkLLMProvider()
    providers = ProviderRegistry(
        llm=llm,
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_depth_limit")
    loop = CompetitionAgentLoop(
        thread_id="thread_depth_limit",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("parent")

    starts = [event for event in monitor.events if event.type == "subagent_started"]
    assert [(event.payload["subagent_id"], event.payload["fork_depth"]) for event in starts] == [
        ("child", 1)
    ]
    child_rows = [
        json.loads(line)
        for line in (tmp_path / "subagents" / "child" / "trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rejected_fork = next(
        observation
        for row in child_rows
        for observation in row["observations"]
        if observation["tool"] == "fork"
    )
    assert [result["task_id"] for result in rejected_fork["subagents"]] == [
        "grandchild_a",
        "grandchild_b",
    ]
    assert {result["status"] for result in rejected_fork["subagents"]} == {"failed"}
    assert all(
        "fork depth 2 exceeds max_fork_depth=1" in result["error"]
        for result in rejected_fork["subagents"]
    )
    assert "Fork is not allowed at this depth." in llm.system_prompts["child"][0]
    assert not (tmp_path / "subagents" / "child" / "subagents").exists()
    assert len([event for event in monitor.events if event.type == "task_result"]) == 1


@pytest.mark.asyncio
async def test_nested_fork_lifecycle_events_keep_current_task_and_depth(tmp_path):
    settings = replace(submission_settings(), max_fork_depth=2)
    base = ProviderRegistry.from_settings(settings)
    providers = ProviderRegistry(
        llm=NestedForkLLMProvider(),
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_nested_events")
    loop = CompetitionAgentLoop(
        thread_id="thread_nested_events",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("parent")

    starts = {
        (event.payload["subagent_id"], event.payload["fork_depth"])
        for event in monitor.events
        if event.type == "subagent_started"
    }
    finishes = {
        (event.payload["task_id"], event.payload["fork_depth"])
        for event in monitor.events
        if event.type == "subagent_finished"
    }
    assert starts == {
        ("child", 1),
        ("grandchild_a", 2),
        ("grandchild_b", 2),
    }
    assert finishes == starts
    assert len([event for event in monitor.events if event.type == "task_started"]) == 1
    assert len([event for event in monitor.events if event.type == "task_result"]) == 1


@pytest.mark.asyncio
async def test_competition_loop_emits_provider_and_ranking_events(tmp_path):
    settings = submission_settings()
    monitor = EventCollector(thread_id="thread_test")
    loop = CompetitionAgentLoop(
        thread_id="thread_test",
        session_dir=tmp_path,
        settings=settings,
        providers=ProviderRegistry.from_settings(settings),
        monitor=monitor,
    )

    summary = await loop.run("我想买一套旅行三件套，预算300，不要塑料")

    event_types = [event.type for event in monitor.events]
    assert "provider_start" in event_types
    assert "provider_end" in event_types
    assert "ranking_decision" in event_types
    assert "task_result" in event_types
    assert summary.products
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "candidates.json").exists()
    assert (tmp_path / "trace.jsonl").exists()


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
    thoughts = [event.payload["action"] for event in monitor.events if event.type == "thought"]
    assert tool_starts == ["plan", "item_search", "rank", "pick"]
    assert thoughts == ["plan", "item_search", "rank", "pick", "finish"]
    assert len(llm.calls) == 6
    assert len(llm.action_calls) == 5
    assert summary.products


@pytest.mark.asyncio
async def test_competition_loop_runs_plan_first_when_llm_skips_it(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    llm = SequenceLLMProvider(
        [
            {"action": "item_search", "arguments": {}, "thought": "Search immediately."},
            {"action": "finish", "message": "Done."},
        ]
    )
    providers = ProviderRegistry(
        llm=llm,
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_skip_plan")
    loop = CompetitionAgentLoop(
        thread_id="thread_skip_plan",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    summary = await loop.run("旅行三件套，预算300，不要塑料")

    tool_starts = [event.tool for event in monitor.events if event.type == "tool_start"]
    thoughts = [event.payload["action"] for event in monitor.events if event.type == "thought"]
    assert tool_starts[0] == "plan"
    assert thoughts[0] == "plan"
    assert summary.message
    assert (tmp_path / "summary.json").exists()


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
    assert [event.type for event in monitor.events if event.type == "tool_start"] == []
    assert "task_result" in [event.type for event in monitor.events]


@pytest.mark.asyncio
async def test_root_terminal_fail_emits_only_root_task_error(tmp_path):
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = ProviderRegistry(
        llm=SequenceLLMProvider(
            [{"action": "fail", "message": "Root could not satisfy objective."}]
        ),
        product=base.product,
        web_search=base.web_search,
        shipping=base.shipping,
    )
    monitor = EventCollector(thread_id="thread_root_fail")
    loop = CompetitionAgentLoop(
        thread_id="thread_root_fail",
        session_dir=tmp_path,
        settings=settings,
        providers=providers,
        monitor=monitor,
    )

    await loop.run("root")

    task_errors = [event for event in monitor.events if event.type == "task_error"]
    assert [event.message for event in task_errors] == ["Root could not satisfy objective."]
    assert task_errors[0].payload == {}
    assert not [event for event in monitor.events if event.type == "subagent_error"]


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
    assert any(observation["tool"] == "AgentPlanner" for observation in rows[0]["observations"])
    provider_events = [event for event in monitor.events if event.type == "provider_end"]
    assert any(event.payload.get("tool") == "AgentPlanner" for event in provider_events)
