import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_process_quantum_success():
    """Test successful quantum circuit processing."""
    with patch("server.services.quantum_processor.execute_quantum_circuit", new=AsyncMock()) as mock_execute:
        mock_execute.return_value = {"statevector": [1, 0]}
        response = client.post(
            "/mcp/quantum",
            json={"qubits": 2, "circuit": {}}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "result": {"statevector": [1, 0]}}

@pytest.mark.asyncio
async def test_process_quantum_failure():
    """Test quantum circuit processing failure."""
    with patch("server.services.quantum_processor.execute_quantum_circuit", new=AsyncMock()) as mock_execute:
        mock_execute.side_effect = Exception("Quantum error")
        response = client.post(
            "/mcp/quantum",
            json={"qubits": 2, "circuit": {}}
        )
        assert response.status_code == 500
        assert "Quantum error" in response.json()["detail"]
