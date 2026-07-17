from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.agent.main_agent import CompetitionAgentLoop
from app.api.connection import ConnectionManager
from app.api.monitor import EventCollector
from app.config import OmniMatchSettings
from app.providers.registry import ProviderRegistry
from app.schemas import AgentEvent, ShoppingQuery, TaskState


app = FastAPI(title="OmniMatch MVP")
manager = ConnectionManager()
TASKS: dict[str, TaskState] = {}
OUTPUT_ROOT = Path("output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/tasks")
async def create_task(request: ShoppingQuery) -> dict:
    thread_id = f"thread_{uuid4().hex[:8]}"
    settings = OmniMatchSettings.from_env()
    provider_modes = settings.provider_modes()
    state = TaskState(
        thread_id=thread_id,
        query=request.query,
        profile=settings.profile,
        provider_modes=provider_modes,
    )
    TASKS[thread_id] = state
    session_dir = OUTPUT_ROOT / thread_id
    asyncio.create_task(_run_task(thread_id, request.query, session_dir, settings))
    return {"thread_id": thread_id, "status": state.status}


@app.get("/api/tasks/{thread_id}")
async def get_task(thread_id: str) -> dict:
    state = TASKS.get(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return state.model_dump(mode="json")


@app.websocket("/ws/{thread_id}")
async def websocket_events(websocket: WebSocket, thread_id: str) -> None:
    state = TASKS.get(thread_id)
    if state is None:
        await websocket.close(code=1008, reason="Task not found")
        return
    await manager.connect(thread_id, websocket)
    for event in state.events:
        await websocket.send_json(event.model_dump(mode="json"))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(thread_id, websocket)


async def _run_task(
    thread_id: str,
    query: str,
    session_dir: Path,
    settings: OmniMatchSettings,
) -> None:
    state = TASKS[thread_id]

    async def sink(event: AgentEvent) -> None:
        await manager.broadcast(event)

    monitor = EventCollector(thread_id=thread_id, sink=sink, events=state.events)
    loop = CompetitionAgentLoop(
        thread_id=thread_id,
        session_dir=session_dir,
        settings=settings,
        providers=ProviderRegistry.from_settings(settings),
        monitor=monitor,
    )
    try:
        summary = await loop.run(query)
        state.status = "completed"
        state.result = summary
        state.trace_paths = {
            "summary": str(session_dir / "summary.json"),
            "candidates": str(session_dir / "candidates.json"),
            "trace": str(session_dir / "trace.jsonl"),
        }
    except Exception as exc:  # pragma: no cover - defensive path exercised manually
        TASKS[thread_id].status = "failed"
        TASKS[thread_id].error = str(exc)
        await monitor.emit("task_error", f"任务失败：{exc}")
