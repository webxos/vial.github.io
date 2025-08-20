from fastapi.testclient import TestClient
from server import app
import asyncio

client = TestClient(app)

def test_quantum_sync():
    request = {"jsonrpc": "2.0", "method": "quantum/sync", "params": {"node_id": "node1", "data": {"test": "data"}}, "id": 1}
    response = client.post("/jsonrpc", json=request)
    assert response.status_code == 200
    result = response.json()
    assert "result" in result
    assert result["result"]["params"]["node_id"] == "node1"

def test_wallet_update():
    request = {"jsonrpc": "2.0", "method": "wallet/update", "params": {"node_id": "node1", "amount": 10.0}, "id": 2}
    response = client.post("/jsonrpc", json=request)
    assert response.status_code == 200
    result = response.json()
    assert "result" in result
    assert result["result"]["status"] == "updated"
    assert result["result"]["balance"] == 10.0
