import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from datetime import datetime, timedelta
import uuid
from fastapi import HTTPException

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_token_generation(client):
    response = await client.post(
        "/alchemist/auth/token",
        data={"username": "test_user", "password": "test_pass"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_invalid_credentials(client):
    response = await client.post(
        "/alchemist/auth/token",
        data={"username": "invalid_user", "password": "wrong_pass"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"

@pytest.mark.asyncio
async def test_protected_endpoint(client):
    token_response = await client.post(
        "/alchemist/auth/token",
        data={"username": "test_user", "password": "test_pass"}
    )
    token = token_response.json()["access_token"]
    response = await client.get("/health", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
