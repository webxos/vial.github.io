from fastapi.testclient import TestClient
from server.mcp_server import app
from unittest.mock import patch

client = TestClient(app)


def auth_token():
    response = client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin"}
    )
    return response.json()["access_token"]


def test_rate_limit():
    with patch("server.services.redis_handler.redis_handler.get") as mock_get, \
         patch("server.services.redis_handler.redis_handler.setex") as mock_setex, \
         patch("server.services.redis_handler.redis_handler.incr") as mock_incr:
        mock_get.return_value = None
        mock_setex.return_value = None
        mock_incr.return_value = 1
        headers = {"Authorization": f"Bearer {auth_token()}"}
        response = client.post(
            "/jsonrpc",
            json={"jsonrpc": "2.0", "method": "status", "id": 1},
            headers=headers
        )
        assert response.status_code == 200


def test_rate_limit_exceeded():
    with patch("server.services.redis_handler.redis_handler.get") as mock_get:
        mock_get.return_value = "100"
        headers = {"Authorization": f"Bearer {auth_token()}"}
        response = client.post(
            "/jsonrpc",
            json={"jsonrpc": "2.0", "method": "status", "id": 1},
            headers=headers
        )
        assert response.status_code == 429
        assert response.json()["detail"] == "Too many requests"
