```python
import pytest
from unittest.mock import patch, AsyncMock
from server.services.llm_router import LLMRouter, LLMRequest
from fastapi import HTTPException

@pytest.mark.asyncio
async def test_llm_router_success():
    """Test successful LLM request routing."""
    router = LLMRouter()
    request = LLMRequest(provider="anthropic", prompt="Test prompt")
    with patch("server.services.llm_router.completion", new=AsyncMock(return_value={"choices": [{"message": {"content": "Test response"}}]})):
        response = await router.route_request(request)
        assert response["choices"][0]["message"]["content"] == "Test response"

@pytest.mark.asyncio
async def test_llm_router_invalid_provider():
    """Test routing to invalid provider."""
    router = LLMRouter()
    request = LLMRequest(provider="invalid", prompt="Test prompt")
    with pytest.raises(HTTPException) as exc:
        await router.route_request(request)
    assert exc.value.status_code == 400
    assert "Invalid provider" in str(exc.value.detail)

@pytest.mark.asyncio
async def test_llm_router_failure():
    """Test LLM routing failure."""
    router = LLMRouter()
    request = LLMRequest(provider="anthropic", prompt="Test prompt")
    with patch("server.services.llm_router.completion", side_effect=Exception("LLM error")):
        with pytest.raises(HTTPException) as exc:
            await router.route_request(request)
        assert exc.value.status_code == 500
        assert "LLM error" in str(exc.value.detail)
```
