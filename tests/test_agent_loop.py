import pytest

from app.agent.main_agent import CompetitionAgentLoop
from app.api.monitor import EventCollector
from app.config import OmniMatchSettings
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
