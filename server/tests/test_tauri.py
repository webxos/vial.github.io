import pytest
from unittest.mock import patch
from server.frontend.tauri_main import execute_mcp_tool

@pytest.mark.asyncio
async def test_execute_mcp_tool_success():
    """Test successful MCP tool execution."""
    state = {"oauth_token": "valid_token"}
    with patch("builtins.state", state):
        result = await execute_mcp_tool("quantum_sync", {}, state)
        assert result == "Executed tool: quantum_sync"

@pytest.mark.asyncio
async def test_execute_mcp_tool_unauthorized():
    """Test unauthorized MCP tool execution."""
    state = {"oauth_token": None}
    with patch("builtins.state", state):
        with pytest.raises(Exception, match="Unauthorized"):
            await execute_mcp_tool("quantum_sync", {}, state)
