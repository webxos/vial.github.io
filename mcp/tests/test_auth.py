from fastapi.testclient import TestClient
from ..main import app

client = TestClient(app)

def test_login():
    response = client.post("/api/auth/token", json={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_protected():
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer dummy_token"})
    assert response.status_code == 401
