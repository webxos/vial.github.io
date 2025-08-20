from fastapi.testclient import TestClient
from ..main import app

client = TestClient(app)

def test_get_wallets():
    response = client.post("/api/auth/token", json={"username": "admin", "password": "admin"})
    token = response.json()["access_token"]
    response = client.get("/api/data/wallets", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
