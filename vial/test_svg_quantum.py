import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.logging_config import logger
import uuid

@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_generate_svg_workflow(client):
    request_id = str(uuid.uuid4())
    workflow_data = {
        "nodes": [{"x": 50, "y": 50, "label": "Quantum Node"}],
        "edges": [{"start": {"x": 50, "y": 100}, "end": {"x": 150, "y": 100}}]
    }
    response = client.post("/v1/visual/generate_workflow", json=workflow_data)
    assert response.status_code == 200
    assert "svg" in response.json()
    logger.info(f"Generate SVG workflow test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_quantum_workflow(client):
    request_id = str(uuid.uuid4())
    task = {"task_name": "quantum_circuit", "params": {"vial_id": "vial1", "qubits": 2}}
    response = client.post("/v1/quantum/execute_circuit", json=task["params"])
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "executed"
    logger.info(f"Quantum workflow test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_svg_quantum_integration(client):
    request_id = str(uuid.uuid4())
    workflow_data = {
        "nodes": [{"x": 50, "y": 50, "label": "Quantum Circuit"}],
        "edges": [{"start": {"x": 50, "y": 100}, "end": {"x": 150, "y": 100}}]
    }
    svg_response = client.post("/v1/visual/generate_workflow", json=workflow_data)
    assert svg_response.status_code == 200
    quantum_response = client.post("/v1/quantum/execute_circuit", json={"vial_id": "vial1", "qubits": 2})
    assert quantum_response.status_code == 200
    logger.info(f"SVG quantum integration test passed", request_id=request_id)
