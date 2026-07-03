from __future__ import annotations

from collections import defaultdict

from fastapi import WebSocket

from app.schemas import AgentEvent


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, thread_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[thread_id].add(websocket)

    def disconnect(self, thread_id: str, websocket: WebSocket) -> None:
        self._connections[thread_id].discard(websocket)
        if not self._connections[thread_id]:
            self._connections.pop(thread_id, None)

    async def broadcast(self, event: AgentEvent) -> None:
        disconnected: list[WebSocket] = []
        for websocket in self._connections.get(event.thread_id, set()):
            try:
                await websocket.send_json(event.model_dump(mode="json"))
            except RuntimeError:
                disconnected.append(websocket)
        for websocket in disconnected:
            self.disconnect(event.thread_id, websocket)
