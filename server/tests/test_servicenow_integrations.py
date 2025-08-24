import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_query_servicenow_success():
    """Test successful ServiceNow API query."""
    with patch("server.services.servicenow_integration.query_servicenow_api", new=AsyncMock()) as mock_query:
        mock_query.return_value = {"result": "ServiceNow response"}
        response = client.post(
            "/mcp/tools/servicenow",
            json={"endpoint": "table/incident", "params": {}}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "result": {"result": "ServiceNow response"}}

@pytest.mark.asyncio
async def test_query_servicenow_failure():
    """Test ServiceNow API query failure."""
    with patch("server.services.servicenow_integration.query_servicenow_api", new=AsyncMock()) as mock_query:
        mock_query.side_effect = Exception("ServiceNow error")
        response = client.post(
            "/mcp/tools/servicenow",
            json={"endpoint": "table/incident", "params": {}}
        )
        assert response.status_code == 500
        assert "ServiceNow error" in response.json()["detail"]
