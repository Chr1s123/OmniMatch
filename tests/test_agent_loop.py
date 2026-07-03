import pytest

from app.agent.main_agent import MockAgentLoop
from app.api.monitor import EventCollector


@pytest.mark.asyncio
async def test_agent_loop_emits_events_and_returns_summary(tmp_path):
    collector = EventCollector(thread_id="thread_test")
    loop = MockAgentLoop(thread_id="thread_test", session_dir=tmp_path, monitor=collector)

    summary = await loop.run("我想买一套旅行三件套，预算300，不要塑料")

    event_types = [event.type for event in collector.events]
    assert "task_started" in event_types
    assert "tool_start" in event_types
    assert "subagent_started" in event_types
    assert "subagent_finished" in event_types
    assert "task_result" in event_types
    assert summary.products
    assert (tmp_path / "summary.json").exists()
