from fastapi.testclient import TestClient

from app.api.server import app


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
