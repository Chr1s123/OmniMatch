from fastapi.testclient import TestClient

from app.api.server import app
from app.api.server import TASKS
from app.schemas import AgentEvent, TaskState


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
