from fastapi.testclient import TestClient

from app.api.server import app
from app.api.server import TASKS
from app.schemas import AgentEvent, TaskState


def test_create_task_returns_thread_id():
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
