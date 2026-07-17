import asyncio
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
    assert {(event.type, event.payload["task_id"], event.payload["status"])
            for event in monitor.events if event.type == "subagent_finished"} == {
        ("subagent_finished", "slow", "timed_out"),
        ("subagent_finished", "fast", "completed"),
    }


@pytest.mark.asyncio
async def test_fork_executor_timeout_includes_semaphore_queue_wait():
    queued_finished = asyncio.Event()

    async def on_event(event) -> None:
        if event.type == "subagent_finished" and event.payload["task_id"] == "queued":
            queued_finished.set()

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
    try:
        await asyncio.wait_for(queued_finished.wait(), timeout=0.2)
    finally:
        release_blocker.set()
        results = await execution

    assert runner_started == {"blocker"}
    assert [(result.task_id, result.status) for result in results] == [
        ("blocker", "completed"),
        ("queued", "timed_out"),
    ]


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
    assert {event.payload["subagent_id"] for event in monitor.events
            if event.type == "subagent_cancelled"} == {"running", "queued_a", "queued_b"}


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
