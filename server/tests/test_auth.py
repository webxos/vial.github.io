from fastapi.testclient import TestClient
from ..mcp_server import app


def test_login_for_access_token():
    client = TestClient(app)
    response = client.post("/auth/token", data={"username": "test", "password": "test"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_invalid_credentials():
    client = TestClient(app)
    response = client.post("/auth/token", data={"username": "wrong", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


def test_get_current_user():
    client = TestClient(app)
    token_response = client.post("/auth/token", data={"username": "test", "password": "test"})
    token = token_response.json()["access_token"]
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["user_id"] == "test"
