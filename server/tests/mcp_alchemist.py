from fastapi.testclient import TestClient
from ..mcp_server import app
from ..services.vial_manager import VialManager


def test_alchemist_auth():
    client = TestClient(app)
    response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_alchemist_train():
    client = TestClient(app)
    token_response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    token = token_response.json()["access_token"]
    response = client.post("/alchemist/train", json={"prompt": "Generate API endpoint"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "trained"
    assert "result" in response.json()


def test_alchemist_wallet_export():
    client = TestClient(app)
    token_response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    token = token_response.json()["access_token"]
    response = client.post("/alchemist/wallet/export", json={"user_id": "test"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "vials" in response.json()


def test_alchemist_troubleshoot():
    client = TestClient(app)
    token_response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    token = token_response.json()["access_token"]
    response = client.post("/alchemist/troubleshoot", json={"error": "Database connection failed"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "troubleshooting"
    assert "options" in response.json()
