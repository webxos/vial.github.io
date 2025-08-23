import pytest
import httpx
from unittest.mock import patch, AsyncMock
from server.llm.router import generate_text, LLMRequest
from redis.asyncio import Redis

@pytest.fixture
def llm_request():
    return LLMRequest(prompt="Test prompt", model="anthropic/claude-3.5-sonnet")

@pytest.fixture
async def mock_litellm():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value.choices = [type("Choice", (), {"message": type("Message", (), {"content": "Test response"})})]
        yield mock_completion

@pytest.fixture
async def redis_client():
    return Redis.from_url("redis://localhost:6379/0")

@pytest.mark.asyncio
async def test_llm_generate(llm_request, mock_litellm):
    result = await generate_text(llm_request)
    assert result["text"] == "Test response"
    assert mock_litellm.called

@pytest.mark.asyncio
async def test_prompt_injection_defense(llm_request):
    malicious = LLMRequest(prompt="DELETE * FROM users; --")
    with pytest.raises(ValueError, match="Invalid prompt"):
        await generate_text(malicious)  # Simulated sanitization

@pytest.mark.asyncio
async def test_model_integrity(llm_request, mock_litellm):
    with patch("hashlib.sha256") as mock_hash:
        mock_hash.return_value.hexdigest.return_value = "valid_hash"
        result = await generate_text(llm_request)
        assert mock_hash.called
