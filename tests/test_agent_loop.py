import json

import pytest

from app.agent.actions import AgentAction
from app.agent.main_agent import CompetitionAgentLoop
from app.api.monitor import EventCollector
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
