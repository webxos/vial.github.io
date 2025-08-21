from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest

client = TestClient(app)

@pytest.fixture
def auth_token():
    response = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    return response.json()["access_token"]

def test_quantum_sync(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post(
        "/jsonrpc",
        json={"jsonrpc": "2.0", "method": "sync", "params": {"node_id": "test_node"}, "id": 1},
        headers=headers
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "synced"
    assert "quantum_state" in response.json()["result"]

def test_quantum_sync_no_node_id(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post(
        "/jsonrpc",
        json={"jsonrpc": "2.0", "method": "sync", "params": {}, "id": 1},
        headers=headers
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Node ID required"
