import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_query_nasa_success():
    """Test successful NASA API query."""
    with patch("server.api.nasa_integration.query_nasa_api", new=AsyncMock()) as mock_query:
        mock_query.return_value = {"data": "NASA response"}
        response = client.post(
            "/mcp/nasa",
            json={"endpoint": "planetary/apod", "params": {}}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "result": {"data": "NASA response"}}

@pytest.mark.asyncio
async def test_query_nasa_failure():
    """Test NASA API query failure."""
    with patch("server.api.nasa_integration.query_nasa_api", new=AsyncMock()) as mock_query:
        mock_query.side_effect = Exception("NASA error")
        response = client.post(
            "/mcp/nasa",
            json={"endpoint": "planetary/apod", "params": {}}
        )
        assert response.status_code == 500
        assert "NASA error" in response.json()["detail"]
