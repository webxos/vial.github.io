import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.api.copilot_integration import CopilotIntegration
from unittest.mock import AsyncMock, patch


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_copilot():
    copilot = CopilotIntegration()
    copilot.git_trainer.get_diff = AsyncMock(return_value="diff content")
    return copilot


@pytest.mark.asyncio
async def test_suggest_code(client, mock_copilot):
    with patch("server.api.copilot_integration.CopilotIntegration", return_value=mock_copilot):
        response = client.post("/copilot/suggest", json={"repo_path": "/tmp/repo", "file_path": "test.py"})
        assert response.status_code == 200
        assert response.json()["suggestion"] == "// Suggested change for test.py: diff content"
        mock_copilot.git_trainer.get_diff.assert_called_once()
