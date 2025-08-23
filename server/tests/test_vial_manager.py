import pytest
import httpx
from unittest.mock import patch, AsyncMock
from server.services.vial_manager import get_vial_status, run_vial_task, VialAgent
from pymongo import MongoClient

@pytest.fixture
def vial_agent():
    return VialAgent(id="vial1", model_type="pytorch", status="idle")

@pytest.fixture
async def mock_mongo():
    with patch("pymongo.MongoClient") as mock_client:
        mock_client.return_value.vial_mcp.agents.find.return_value = [{"id": "vial1", "model_type": "pytorch", "status": "idle"}]
        mock_client.return_value.vial_mcp.agents.find_one.return_value = {"id": "vial1", "model_type": "pytorch", "status": "idle"}
        yield mock_client

@pytest.mark.asyncio
async def test_vial_status(mock_mongo):
    result = await get_vial_status()
    assert len(result) > 0
    assert result[0]["id"] == "vial1"

@pytest.mark.asyncio
async def test_vial_run_task(vial_agent, mock_mongo):
    with patch("torch.nn.Linear") as mock_model:
        mock_model.return_value.return_value.item.return_value = 0.5
        result = await run_vial_task(vial_agent.id, "test_task")
        assert "output" in result
        assert mock_mongo.called

@pytest.mark.asyncio
async def test_agent_sandboxing(vial_agent):
    with pytest.raises(ValueError, match="Invalid task"):
        await run_vial_task(vial_agent.id, "rm -rf /")  # Simulated sandbox
