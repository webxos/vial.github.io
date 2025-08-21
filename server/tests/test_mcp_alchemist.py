import pytest
import torch
from server.models.mcp_alchemist import MCPAlchemist
from server.mcp_server import app
from fastapi.testclient import TestClient


client = TestClient(app)


@pytest.mark.asyncio
async def test_predict_quantum_outcome():
    alchemist = MCPAlchemist()
    circuit = {
        "qubits": 2,
        "gates": [{"type": "h", "target": 0}, {"type": "cx", "control": 0, "target": 1}]
    }
    response = await alchemist.predict_quantum_outcome(circuit)
    assert "prediction" in response
    assert isinstance(response["prediction"], list)
    assert len(response["prediction"]) == 8
