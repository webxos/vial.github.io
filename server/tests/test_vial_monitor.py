from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest

client = TestClient(app)

@pytest.fixture
def auth_token():
    response = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    return response.json()["access_token"]

def test_monitor_endpoint(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post(
        "/jsonrpc",
        json={"jsonrpc": "2.0", "method": "monitor", "id": 1},
        headers=headers
    )
    assert response.status_code == 200
    assert "cpu_usage" in response.json()["result"]
    assert "memory_usage" in response.json()["result"]
    assert "disk_usage" in response.json()["result"]
    assert "containers" in response.json()["result"]
