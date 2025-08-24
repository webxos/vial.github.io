import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_quantum_rag_success():
    """Test successful quantum-RAG query."""
    with patch("server.services.quantum_rag.process_quantum_rag", new=AsyncMock()) as mock_process:
        mock_process.return_value = {"statevector": [1, 0], "rag_results": [{"doc": "test"}]}
        response = client.post(
            "/mcp/quantum_rag",
            json={"query": "test query", "qubits": 2, "max_results": 5}
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "result": {"statevector": [1, 0], "rag_results": [{"doc": "test"}]}
        }

@pytest.mark.asyncio
async def test_quantum_rag_failure():
    """Test quantum-RAG query failure."""
    with patch("server.services.quantum_rag.process_quantum_rag", new=AsyncMock()) as mock_process:
        mock_process.side_effect = Exception("Quantum-RAG error")
        response = client.post(
            "/mcp/quantum_rag",
            json={"query": "test query", "qubits": 2, "max_results": 5}
        )
        assert response.status_code == 500
        assert "Quantum-RAG error" in response.json()["detail"]
