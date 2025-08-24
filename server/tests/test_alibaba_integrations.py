import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_query_alibaba_success():
    """Test successful Alibaba Cloud API query."""
    with patch("server.api.alibaba_integration.query_alibaba_api", new=AsyncMock()) as mock_query:
        mock_query.return_value = {"result": "Alibaba response"}
        response = client.post(
            "/mcp/tools/alibaba",
            json={"endpoint": "dataworks", "params": {}}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "result": {"result": "Alibaba response"}}

@pytest.mark.asyncio
async def test_query_alibaba_failure():
    """Test Alibaba Cloud API query failure."""
    with patch("server.api.alibaba_integration.query_alibaba_api", new=AsyncMock()) as mock_query:
        mock_query.side_effect = Exception("Alibaba error")
        response = client.post(
            "/mcp/tools/alibaba",
            json={"endpoint": "dataworks", "params": {}}
        )
        assert response.status_code == 500
        assert "Alibaba error" in response.json()["detail"]
