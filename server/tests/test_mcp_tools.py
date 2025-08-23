import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from server.api.mcp_tools import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_mcp_tools(client):
    with patch("server.validation.mcp_validator.MCPValidator.validate_tool_call", new_callable=AsyncMock) as mock_validate:
        mock_validate.return_value = True
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_shield:
            mock_shield.return_value.json.return_value = {"malicious": False}
            response = client.get("/mcp/tools", headers={"Authorization": "Bearer mock_token"})
            assert response.status_code == 200
            assert "tools" in response.json()
            assert len(response.json()["tools"]) >= 4

@pytest.mark.asyncio
async def test_tools_prompt_shield(client):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_shield:
        mock_shield.return_value.json.return_value = {"malicious": True}
        response = client.get("/mcp/tools", headers={"Authorization": "Bearer mock_token"})
        assert response.status_code == 400
        assert "Malicious input detected" in response.json()["detail"]
