import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.main_agent import MockAgentLoop
from app.api.monitor import EventCollector


async def main() -> None:
    thread_id = "thread_example"
    monitor = EventCollector(thread_id=thread_id)
    loop = MockAgentLoop(
        thread_id=thread_id,
        session_dir=Path("output") / thread_id,
        monitor=monitor,
    )
    summary = await loop.run("我想买一套旅行三件套，预算300，不要塑料")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
