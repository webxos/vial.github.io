from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_execute_task(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    response = client.post("/agent/execute-task", json={"vial_id": "vial1", "task": "test_task"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["vial_id"] == "vial1"
    assert "output" in response.json()
