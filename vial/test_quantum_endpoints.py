import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.logging_config import logger
import uuid

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_quantum_circuit_endpoint(client):
    request_id = str(uuid.uuid4())
    response = client.post(
        "/v1/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "quantum_circuit",
            "params": {"qubits": 4, "network_id": "54965687-3871-4f3d-a803-ac9840af87c4"},
            "id": "1"
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "circuit_built"
    assert "circuit" in response.json()["result"]
    logger.info("Quantum circuit endpoint test passed", request_id=request_id)

@pytest.mark.asyncio
async def test_jsonrpc_handler(client):
    request_id = str(uuid.uuid4())
    response = client.post(
        "/v1/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "agent_coord",
            "params": {"network_id": "54965687-3871-4f3d-a803-ac9840af87c4"},
            "id": "1"
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "coordinated"
    assert len(response.json()["result"]["results"]) == 4
    logger.info("JSON-RPC handler test passed", request_id=request_id)
