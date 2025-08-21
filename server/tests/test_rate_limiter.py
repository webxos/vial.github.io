from fastapi.testclient import TestClient
from server.mcp_server import app
from unittest.mock import patch
import pytest

client = TestClient(app)

@pytest.fixture
def auth_token():
    response = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    return response.json()["access_token"]

@patch("server.services.redis_handler.redis_handler.get")
@patch("server.services.redis_handler.redis_handler.setex")
@patch("server.services.redis_handler.redis_handler.incr")
def test_rate_limit(mock_incr, mock_setex, mock_get, auth_token):
    mock_get.return_value = None
    mock_setex.return_value = None
    mock_incr.return_value = 1
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": "status", "id": 1}, headers=headers)
    assert response.status_code == 200

@patch("server.services.redis_handler.redis_handler.get")
def test_rate_limit_exceeded(mock_get, auth_token):
    mock_get.return_value = "100"
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": "status", "id": 1}, headers=headers)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == 429
