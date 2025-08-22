import pytest
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials, HTTPException
from server.mcp_server import app
from server.api.security import SecurityManager


@pytest.fixture
def client():
    return TestClient(app)


def test_security_headers():
    security = SecurityManager()
    headers = security.get_security_headers()
    assert "X-Content-Type-Options" in headers
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_authenticate_valid():
    security = SecurityManager()
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
    result = security.authenticate(credentials)
    assert result is True


def test_authenticate_invalid():
    security = SecurityManager()
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid_token")
    try:
        security.authenticate(credentials)
    except HTTPException as e:
        assert e.status_code == 401
