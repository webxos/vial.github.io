from fastapi.testclient import TestClient
from server.mcp_server import app

client = TestClient(app)


def test_jsonrpc_compliance():
    response = client.post("/jsonrpc", json={
        "jsonrpc": "2.0",
        "method": "status",
        "id": 1
    }, headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 200
    assert response.json()["jsonrpc"] == "2.0"


def test_invalid_jsonrpc_version():
    response = client.post("/jsonrpc", json={
        "jsonrpc": "1.0",
        "method": "status",
        "id": 1
    }, headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 400
    assert "Invalid JSON-RPC version" in response.json()["detail"]


def test_method_not_found():
    response = client.post("/jsonrpc", json={
        "jsonrpc": "2.0",
        "method": "invalid",
        "id": 1
    }, headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 400
    assert "Method not found" in response.json()["detail"]
