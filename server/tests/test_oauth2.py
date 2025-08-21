from fastapi.testclient import TestClient
from server.mcp_server import app

client = TestClient(app)


def test_token_endpoint():
    response = client.post("/auth/token",
                          data={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_invalid_credentials():
    response = client.post("/auth/token",
                          data={"username": "invalid", "password": "invalid"})
    assert response.status_code == 401
    assert "Invalid credentials" in response.json()["detail"]


def test_missing_credentials():
    response = client.post("/auth/token", data={})
    assert response.status_code == 400
    assert "Missing credentials" in response.json()["detail"]


def test_generate_credentials():
    response = client.post("/auth/generate-credentials",
                          headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 200
    assert "key" in response.json()
    assert "secret" in response.json()
