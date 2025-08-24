```python
import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_quantum_rag_success():
    """Test successful quantum-RAG query."""
    with patch("server.services.quantum_processor.QuantumProcessor.execute_quantum_rag", new=AsyncMock(return_value=["result1", "result2"])):
        response = client.post("/mcp/quantum_rag", json={"query": "test query", "quantum_circuit": "H 0; CX 0 1", "max_results": 5})
        assert response.status_code == 200
        assert response.json() == {"results": ["result1", "result2"]}

@pytest.mark.asyncio
async def test_quantum_rag_failure():
    """Test quantum-RAG endpoint failure."""
    with patch("server.services.quantum_processor.QuantumProcessor.execute_quantum_rag", side_effect=Exception("Quantum error")):
        response = client.post("/mcp/quantum_rag", json={"query": "test query", "quantum_circuit": "H 0; CX 0 1", "max_results": 5})
        assert response.status_code == 500
        assert "Quantum error" in response.json()["detail"]
```
