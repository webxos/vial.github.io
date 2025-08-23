import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_route_llm_success():
    """Test successful LLM routing."""
    with patch("server.services.llm_router.route_to_provider", new=AsyncMock()) as mock_route:
        mock_route.return_value = {"response": "LLM output"}
        response = client.post(
            "/mcp/llm",
            json={"provider": "anthropic", "prompt": "Hello"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "result": {"response": "LLM output"}}

@pytest.mark.asyncio
async def test_route_llm_invalid_provider():
    """Test invalid LLM provider error."""
    with patch("server.services.llm_router.route_to_provider", new=AsyncMock()) as mock_route:
        mock_route.side_effect = HTTPException(status_code=400, detail="Invalid provider")
        response = client.post(
            "/mcp/llm",
            json={"provider": "invalid", "prompt": "Hello"}
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid provider"}
