import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def memory_manager():
    return MemoryManager()

@pytest.mark.asyncio
async def test_svg_task_quantum(client, memory_manager):
    request_id = str(uuid.uuid4())
    response = client.post(
        "/v1/execute_svg_task",
        json={
            "task_name": "create_endpoint",
            "params": {
                "vial_id": "vial1",
                "x_position": 0,
                "y_position": 0,
                "endpoint": "/v1/custom/test_endpoint",
                "quantum_logic": {"qubits": 4, "gates": ["H", "CNOT"]}
            }
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    quantum_logic = await memory_manager.get_quantum_logic("create_endpoint", request_id)
    assert quantum_logic["qubits"] == 4
    assert quantum_logic["gates"] == ["H", "CNOT"]
    logger.info("SVG task with quantum logic test passed", request_id=request_id)

@pytest.mark.asyncio
async def test_quantum_circuit(client):
    request_id = str(uuid.uuid4())
    response = client.post(
        "/v1/quantum_circuit",
        json={"qubits": 4, "network_id": "54965687-3871-4f3d-a803-ac9840af87c4"},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "circuit_built"
    assert "circuit" in response.json()
    logger.info("Quantum circuit test passed", request_id=request_id)
