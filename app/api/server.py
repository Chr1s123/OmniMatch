from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.agent.main_agent import MockAgentLoop
from app.api.connection import ConnectionManager
from app.api.monitor import EventCollector
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
    state = TaskState(thread_id=thread_id, query=request.query)
    TASKS[thread_id] = state
    session_dir = OUTPUT_ROOT / thread_id
    asyncio.create_task(_run_task(thread_id, request.query, session_dir))
    return {"thread_id": thread_id, "status": state.status}


@app.get("/api/tasks/{thread_id}")
async def get_task(thread_id: str) -> dict:
    state = TASKS.get(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return state.model_dump(mode="json")


@app.websocket("/ws/{thread_id}")
async def websocket_events(websocket: WebSocket, thread_id: str) -> None:
    await manager.connect(thread_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(thread_id, websocket)


async def _run_task(thread_id: str, query: str, session_dir: Path) -> None:
    async def sink(event: AgentEvent) -> None:
        state = TASKS[thread_id]
        state.events.append(event)
        await manager.broadcast(event)

    monitor = EventCollector(thread_id=thread_id, sink=sink)
    loop = MockAgentLoop(thread_id=thread_id, session_dir=session_dir, monitor=monitor)
    try:
        summary = await loop.run(query)
        TASKS[thread_id].status = "completed"
        TASKS[thread_id].result = summary
    except Exception as exc:  # pragma: no cover - defensive path exercised manually
        TASKS[thread_id].status = "failed"
        TASKS[thread_id].error = str(exc)
        await monitor.emit("task_error", f"任务失败：{exc}")
