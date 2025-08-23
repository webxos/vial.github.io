import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_execute_plugin_success():
    """Test successful plugin execution."""
    with patch("server.plugins.plugin_manager.PluginManager.execute_plugin", new=AsyncMock()) as mock_execute:
        mock_execute.return_value = {"result": "success"}
        response = client.post(
            "/mcp/plugins",
            json={"plugin_name": "quantum_sync", "params": {}}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "result": {"result": "success"}}

@pytest.mark.asyncio
async def test_execute_plugin_not_found():
    """Test plugin not found error."""
    with patch("server.plugins.plugin_manager.PluginManager.execute_plugin", new=AsyncMock()) as mock_execute:
        mock_execute.side_effect = HTTPException(status_code=404, detail="Plugin not found")
        response = client.post(
            "/mcp/plugins",
            json={"plugin_name": "invalid_plugin", "params": {}}
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "Plugin not found"}
