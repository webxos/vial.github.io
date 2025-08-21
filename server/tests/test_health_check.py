from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest

client = TestClient(app)

@pytest.fixture
def auth_token():
    response = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    return response.json()["access_token"]

def test_health_check(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post(
        "/jsonrpc",
        json={"jsonrpc": "2.0", "method": "health_check", "id": 1},
        headers=headers
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "healthy"
    assert "services" in response.json()["result"]
    assert all(key in response.json()["result"]["services"] for key in ["mongodb", "redis", "sqlite"])
