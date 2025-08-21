from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest

client = TestClient(app)

def test_oauth_token():
    response = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_oauth_invalid_credentials():
    response = client.post("/auth/token", data={"username": "invalid", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"

def test_protected_endpoint():
    response = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": "status", "id": 1}, headers=headers)
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "running"
