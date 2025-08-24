import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_rewards_success():
    """Test successful rewards processing."""
    with patch("server.services.webxos_rewards.calculate_rewards", new=AsyncMock()) as mock_rewards:
        mock_rewards.return_value = {"status": "success", "rewards": {"points": 100, "action": "test_action"}}
        response = client.post(
            "/mcp/rewards",
            json={"user_id": "user123", "action": "test_action", "metadata": {}}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "rewards": {"points": 100, "action": "test_action"}}

@pytest.mark.asyncio
async def test_rewards_failure():
    """Test rewards processing failure."""
    with patch("server.services.webxos_rewards.calculate_rewards", new=AsyncMock()) as mock_rewards:
        mock_rewards.side_effect = Exception("Rewards error")
        response = client.post(
            "/mcp/rewards",
            json={"user_id": "user123", "action": "test_action", "metadata": {}}
        )
        assert response.status_code == 500
        assert "Rewards error" in response.json()["detail"]
