import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.logging_config import logger
import uuid

@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_auth_integration(client):
    request_id = str(uuid.uuid4())
    response = client.post("/v1/auth/token", json={"network_id": "test_net", "session_id": "test_sess"})
    assert response.status_code == 200
    assert "token" in response.json()
    logger.info(f"Auth integration test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_wallet_crud_integration(client):
    request_id = str(uuid.uuid4())
    wallet_data = {"network_id": "test_net", "balance": 100}
    create_response = client.post("/v1/crud/create_wallet", json=wallet_data)
    assert create_response.status_code == 200
    wallet_id = create_response.json()["wallet_id"]
    get_response = client.get(f"/v1/crud/get_wallet/{wallet_id}")
    assert get_response.status_code == 200
    assert get_response.json()["wallet"]["network_id"] == "test_net"
    logger.info(f"Wallet CRUD integration test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_quantum_integration(client):
    request_id = str(uuid.uuid4())
    params = {"vial_id": "vial1", "qubits": 2}
    response = client.post("/v1/quantum/execute_circuit", json=params)
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "executed"
    logger.info(f"Quantum integration test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_workflow_integration(client):
    request_id = str(uuid.uuid4())
    workflow_data = {
        "nodes": [{"x": 50, "y": 50, "label": "Researcher"}],
        "edges": [{"start": {"x": 50, "y": 100}, "end": {"x": 150, "y": 100}}]
    }
    response = client.post("/v1/visual/generate_workflow", json=workflow_data)
    assert response.status_code == 200
    assert "svg" in response.json()
    logger.info(f"Workflow integration test passed", request_id=request_id)
