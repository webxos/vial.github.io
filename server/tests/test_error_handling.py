import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_http_exception_handling(client):
    response = client.get("/nonexistent")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_general_exception_handling(client):
    response = client.post("/agent/task", json={"task_id": "invalid_task"})
    assert response.status_code == 500
    assert "detail" in response.json()
