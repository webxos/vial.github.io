import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.logging_config import logger
import uuid

@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_execute_quantum_circuit(client):
    request_id = str(uuid.uuid4())
    params = {"vial_id": "vial1", "qubits": 2}
    response = client.post("/v1/quantum/execute_circuit", json=params)
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "executed"
    assert "circuit_id" in response.json()["result"]
    logger.info(f"Execute quantum circuit test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_get_circuit_status(client):
    request_id = str(uuid.uuid4())
    circuit_id = str(uuid.uuid4())
    response = client.get(f"/v1/quantum/circuit_status/{circuit_id}")
    assert response.status_code == 200
    assert response.json()["status"]["circuit_id"] == circuit_id
    logger.info(f"Get circuit status test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_invalid_quantum_params(client):
    request_id = str(uuid.uuid4())
    params = {"vial_id": "vial1", "qubits": -1}
    response = client.post("/v1/quantum/execute_circuit", json=params)
    assert response.status_code == 500
    logger.info(f"Invalid quantum params test passed", request_id=request_id)
