from fastapi.testclient import TestClient
from server.mcp_server import app
from unittest.mock import Mock

client = TestClient(app)


def auth_token():
    response = client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin"}
    )
    return response.json()["access_token"]


def test_rate_limit():
    mock_get = Mock(return_value=None)
    mock_setex = Mock(return_value=None)
    mock_incr = Mock(return_value=1)
    with patch("server.services.redis_handler.redis_handler.get", mock_get), \
         patch("server.services.redis_handler.redis_handler.setex", mock_setex), \
         patch("server.services.redis_handler.redis_handler.incr", mock_incr):
        headers = {"Authorization": f"Bearer {auth_token()}"}
        response = client.post(
            "/jsonrpc",
            json={"jsonrpc": "2.0", "method": "status", "id": 1},
            headers=headers
        )
        assert response.status_code == 200


def test_rate_limit_exceeded():
    mock_get = Mock(return_value="100")
    with patch("server.services.redis_handler.redis_handler.get", mock_get):
        headers = {"Authorization": f"Bearer {auth_token()}"}
        response = client.post(
            "/jsonrpc",
            json={"jsonrpc": "2.0", "method": "status", "id": 1},
            headers=headers
        )
        assert response.status_code == 429
        assert response.json()["detail"] == "Too many requests"
