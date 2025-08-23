import pytest
from unittest.mock import patch, AsyncMock
from server.services.llm_router import LLMRouter, LLMRequest

@pytest.fixture
def router():
    return LLMRouter()

@pytest.mark.asyncio
async def test_llm_routing(router):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [
            AsyncMock(json=lambda: {"malicious": False}),
            AsyncMock(json=lambda: {"text": "Test response"})
        ]
        result = await router.route_request("anthropic", "Test prompt")
        assert "text" in result
        assert result["text"] == "Test response"

@pytest.mark.asyncio
async def test_prompt_shield(router):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_shield:
        mock_shield.return_value.json.return_value = {"malicious": True}
        with pytest.raises(ValueError, match="Malicious input detected"):
            await router.route_request("anthropic", "DROP TABLE users")
