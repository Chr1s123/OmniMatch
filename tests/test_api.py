import pytest
from fastapi.testclient import TestClient

import app.api.server as server
from app.api.monitor import EventCollector
from app.api.server import app, TASKS
from app.config import OmniMatchSettings
from app.schemas import AgentEvent, TaskState


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


def test_create_task_returns_thread_id(monkeypatch):
    monkeypatch.setenv("OMNIMATCH_PROFILE", "submission")
    client = TestClient(app)
    response = client.post("/api/tasks", json={"query": "旅行三件套，预算300"})
    assert response.status_code == 200
    data = response.json()
    assert data["thread_id"].startswith("thread_")
    assert data["status"] == "running"


def test_get_unknown_task_returns_404():
    client = TestClient(app)
    response = client.get("/api/tasks/thread_missing")
    assert response.status_code == 404


def test_websocket_replays_existing_events():
    client = TestClient(app)
    event = AgentEvent(
        type="task_result",
        thread_id="thread_replay",
        run_id="run_replay",
        message="购物清单已生成。",
        payload={"summary": {"message": "ok", "products": [], "warnings": []}},
    )
    TASKS["thread_replay"] = TaskState(
        thread_id="thread_replay",
        status="completed",
        events=[event],
    )

    with client.websocket_connect("/ws/thread_replay") as websocket:
        data = websocket.receive_json()

    assert data["type"] == "task_result"
    assert data["payload"]["summary"]["message"] == "ok"


def test_get_task_includes_profile_and_trace_paths():
    client = TestClient(app)
    TASKS["thread_done"] = TaskState(
        thread_id="thread_done",
        status="completed",
        profile="submission",
        provider_modes={"llm": "placeholder"},
        trace_paths={"summary": "output/thread_done/summary.json"},
    )

    response = client.get("/api/tasks/thread_done")

    assert response.status_code == 200
    data = response.json()
    assert data["profile"] == "submission"
    assert data["provider_modes"]["llm"] == "placeholder"
    assert data["trace_paths"]["summary"].endswith("summary.json")


def test_unknown_websocket_thread_is_rejected():
    client = TestClient(app)

    try:
        with client.websocket_connect("/ws/thread_missing"):
            raise AssertionError("unknown websocket should not stay connected")
    except Exception as exc:
        assert "1008" in str(exc) or "WebSocketDisconnect" in exc.__class__.__name__


@pytest.mark.asyncio
async def test_run_task_uses_task_state_events_as_single_canonical_store(monkeypatch, tmp_path):
    thread_id = "thread_canonical_events"
    state = TaskState(thread_id=thread_id, query="旅行三件套，预算300")
    TASKS[thread_id] = state
    collectors: list[EventCollector] = []
    broadcast_events: list[AgentEvent] = []

    class CapturingEventCollector(EventCollector):
        def __init__(self, thread_id, sink=None, events=None):
            super().__init__(thread_id, sink=sink, events=events)
            collectors.append(self)

    async def record_broadcast(event: AgentEvent) -> None:
        broadcast_events.append(event)

    monkeypatch.setattr(server, "EventCollector", CapturingEventCollector)
    monkeypatch.setattr(server.manager, "broadcast", record_broadcast)

    try:
        await server._run_task(
            thread_id,
            state.query or "",
            tmp_path / thread_id,
            submission_settings(),
        )
    finally:
        TASKS.pop(thread_id, None)

    assert len(collectors) == 1
    assert collectors[0].events is state.events
    assert len(state.events) == len(broadcast_events)
    assert all(replayed is delivered for replayed, delivered in zip(state.events, broadcast_events))
    assert len({id(event) for event in state.events}) == len(state.events)
    assert state.status == "completed"
