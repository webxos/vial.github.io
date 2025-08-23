import pytest
from unittest.mock import patch, AsyncMock
import torch
from server.models.mcp_alchemist import MCPAlchemist, AlchemistTask

@pytest.fixture
def alchemist():
    return MCPAlchemist()

@pytest.mark.asyncio
async def test_quantum_processing(alchemist):
    with patch("torch.tensor") as mock_tensor:
        mock_tensor.return_value.tolist.return_value = [0.5]
        task = AlchemistTask(task_id="task1", input_data={"value": 0.5}, llm_provider="anthropic")
        result = await alchemist.process_task(task)
        assert "quantum_result" in result
        assert result["quantum_result"] == [0.5]

@pytest.mark.asyncio
async def test_llm_routing(alchemist):
    with patch("server.services.llm_router.LLMRouter.route_request", new_callable=AsyncMock) as mock_route:
        mock_route.return_value = {"text": "LLM response"}
        task = AlchemistTask(task_id="task2", input_data={"value": 0.5}, llm_provider="mistral")
        result = await alchemist.process_task(task)
        assert "llm_result" in result
        assert result["llm_result"] == "LLM response"

@pytest.mark.asyncio
async def test_prompt_shield(alchemist):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_shield:
        mock_shield.return_value.json.return_value = {"malicious": True}
        task = AlchemistTask(task_id="task3", input_data={"value": "DROP TABLE users"})
        with pytest.raises(ValueError, match="Malicious input detected"):
            await alchemist.process_task(task)
