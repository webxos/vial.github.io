from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_performance_tuner(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    response = client.post("/optimize", json={"endpoint": "/health", "params": {"key": "value"}}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "optimized"
    assert response.json()["params"] == {"key": "value"}
