import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_competition_agent import main as run_competition_main


async def main() -> None:
    os.environ.setdefault("OMNIMATCH_PROFILE", "dev")
    await run_competition_main()


if __name__ == "__main__":
    asyncio.run(main())
