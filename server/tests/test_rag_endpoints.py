import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_rag_query_success():
    """Test successful RAG query."""
    with patch("server.api.rag_endpoint.perform_vector_search", new=AsyncMock()) as mock_search, \
         patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_llm:
        mock_search.return_value = [{"doc": "test"}]
        mock_llm.return_value.json.return_value = {"response": "LLM output"}
        response = client.post(
            "/mcp/rag",
            json={"query": "test query", "max_results": 5}
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "results": [{"doc": "test"}],
            "llm_response": {"response": "LLM output"}
        }

@pytest.mark.asyncio
async def test_rag_query_failure():
    """Test RAG query failure."""
    with patch("server.api.rag_endpoint.perform_vector_search", new=AsyncMock()) as mock_search:
        mock_search.side_effect = Exception("Vector search error")
        response = client.post(
            "/mcp/rag",
            json={"query": "test query", "max_results": 5}
        )
        assert response.status_code == 500
        assert "Vector search error" in response.json()["detail"]
