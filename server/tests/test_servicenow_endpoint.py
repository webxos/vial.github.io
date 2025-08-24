```python
import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_servicenow_ticket_success():
    """Test successful ServiceNow ticket creation."""
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=AsyncMock(status_code=201, json=lambda: {"result": {"number": "INC001"}}))):
        response = client.post(
            "/mcp/servicenow/ticket",
            json={"short_description": "Test ticket", "description": "Test description", "urgency": "low"}
        )
        assert response.status_code == 200
        assert response.json()["result"]["number"] == "INC001"

@pytest.mark.asyncio
async def test_servicenow_ticket_failure():
    """Test ServiceNow ticket creation failure."""
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.HTTPStatusError(
        message="Unauthorized", request=AsyncMock(), response=AsyncMock(status_code=401)
    ))):
        response = client.post(
            "/mcp/servicenow/ticket",
            json={"short_description": "Test ticket", "description": "Test description", "urgency": "low"}
        )
        assert response.status_code == 401
        assert "Unauthorized" in response.json()["detail"]
```
