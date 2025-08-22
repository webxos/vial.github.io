from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_success(client: TestClient):
    response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_auth_failure(client: TestClient):
    response = client.post("/auth/token", data={"username": "admin", "password": "wrong"})
    assert response.status_code == 401


def test_rbac_admin_access(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    token = token_response.json()["access_token"]
    response = client.get("/verify", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
