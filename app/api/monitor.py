from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from app.schemas import AgentEvent


EventSink = Callable[[AgentEvent], Awaitable[None]]


class EventCollector:
    def __init__(self, thread_id: str, sink: EventSink | None = None) -> None:
        self.thread_id = thread_id
        self.run_id = f"run_{uuid4().hex[:8]}"
        self.events: list[AgentEvent] = []
        self._sink = sink

    async def emit(
        self,
        event_type: str,
        message: str,
        tool: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        event = AgentEvent(
            type=event_type,
            thread_id=self.thread_id,
            run_id=self.run_id,
            tool=tool,
            message=message,
            payload=payload or {},
        )
        self.events.append(event)
        if self._sink:
            await self._sink(event)
        return event
