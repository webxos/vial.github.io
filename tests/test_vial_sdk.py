from fastapi.testclient import TestClient
from server import app
from server.security.oauth2 import create_access_token

client = TestClient(app)

def test_sdk_initialize():
    token = create_access_token({"sub": "admin"})
    response = client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": "sdk/initialize", "id": 1}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    result = response.json()
    assert "result" in result
    assert result["result"]["status"] == "initialized"

def test_sdk_train_and_sync():
    token = create_access_token({"sub": "admin"})
    response = client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": "sdk/train_and_sync", "params": {"data": "1,2,3,4,5"}, "id": 2}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    result = response.json()
    assert "result" in result
    assert result["result"]["status"] == "trained"
