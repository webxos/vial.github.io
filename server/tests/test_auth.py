import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.mcp.auth import map_oauth_to_mcp_session
from jose import jwt
from server.config import settings
from datetime import datetime
from uuid import uuid4 as _uuid4
from fastapi import HTTPException


@pytest.fixture
def client():
    return TestClient(app)


def test_map_oauth_to_mcp_session():
    token_data = {
        "sub": "test_user",
        "scopes": ["wallet:read", "vial:train"],
        "exp": int(datetime.utcnow().timestamp()) + 3600
    }
    token = jwt.encode(token_data, settings.JWT_SECRET, algorithm="HS256")
    request_id = str(_uuid4())
    session = map_oauth_to_mcp_session(token, request_id)
    assert session["user_id"] == "test_user"
    assert "session_id" in session
    assert session["scopes"] == ["wallet:read", "vial:train"]


def test_map_oauth_invalid_token():
    request_id = str(_uuid4())
    with pytest.raises(HTTPException) as exc:
        map_oauth_to_mcp_session("invalid_token", request_id)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid token"


def test_auth_token_endpoint(client):
    response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    assert response.status_code == 401  # No user in test DB
    assert response.json()["detail"] == "Invalid credentials"
