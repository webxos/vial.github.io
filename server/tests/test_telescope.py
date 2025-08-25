import pytest
import asyncio
from fastapi.testclient import TestClient
from server.main import app
from server.services.telescope_service import TelescopeService
from unittest.mock import AsyncMock, patch
import os

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_stream_gibs(client):
    with patch.object(TelescopeService, 'stream_gibs_image', new=AsyncMock(return_value=b"mock_image_data")):
        response = client.get("/mcp/telescope/gibs?date=2023-01-01", headers={"Authorization": "Bearer mock_token"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

@pytest.mark.asyncio
async def test_get_apod(client):
    mock_apod_data = {"title": "Test APOD", "url": "https://example.com/image.jpg"}
    with patch.object(TelescopeService, 'get_apod', new=AsyncMock(return_value=mock_apod_data)):
        response = client.get("/mcp/telescope/apod?date=2023-01-01", headers={"Authorization": "Bearer mock_token"})
        assert response.status_code == 200
        assert response.json() == mock_apod_data

@pytest.mark.asyncio
async def test_gibs_unauthorized(client):
    response = client.get("/mcp/telescope/gibs?date=2023-01-01")
    assert response.status_code == 401

if __name__ == "__main__":
    pytest.main(["-v"])
