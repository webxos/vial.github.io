from fastapi.testclient import TestClient
from server import app
from server.security.oauth2 import create_access_token

client = TestClient(app)

def test_monitor_health():
    token = create_access_token({"sub": "admin"})
    response = client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": "monitor/health", "id": 1}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    result = response.json()
    assert "result" in result
    assert "cpu_usage" in result["result"]
    assert "memory_usage" in result["result"]

def test_monitor_training():
    token = create_access_token({"sub": "admin"})
    response = client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": "monitor/training", "id": 2}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    result = response.json()
    assert "result" in result
    assert "status" in result["result"]
