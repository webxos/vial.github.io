from fastapi.testclient import TestClient
from ..main import app

client = TestClient(app)

def test_create_wallet():
    response = client.post("/api/auth/token", json={"username": "admin", "password": "admin"})
    token = response.json()["access_token"]
    response = client.post("/api/wallet/create", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "wallet_id" in response.json()
