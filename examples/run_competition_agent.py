import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.main_agent import CompetitionAgentLoop
from app.api.monitor import EventCollector
from app.config import OmniMatchSettings
from app.providers.registry import ProviderRegistry


async def main() -> None:
    settings = OmniMatchSettings.from_env()
    thread_id = "thread_example"
    monitor = EventCollector(thread_id=thread_id)
    loop = CompetitionAgentLoop(
        thread_id=thread_id,
        session_dir=Path("output") / thread_id,
        settings=settings,
        providers=ProviderRegistry.from_settings(settings),
        monitor=monitor,
    )
    summary = await loop.run("我想买一套旅行三件套，预算300，不要塑料")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
