import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.mcp_alchemist import Alchemist

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_vial_status_integration(client):
    response = await client.post(
        "/mcp/vial_status_get",
        json={"vial_id": "test_vial"},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json().get("vial_id") == "test_vial"
    assert "balance" in response.json()

@pytest.mark.asyncio
async def test_quantum_circuit_integration(client):
    response = await client.post(
        "/quantum/circuit",
        json={"qubits": 2, "gates": ["h", "cx"]},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "circuit" in response.json()

@pytest.mark.asyncio
async def test_wallet_reward_integration(client):
    response = await client.post(
        "/wallet/reward",
        json={"vial_id": "test_vial", "amount": 10.0},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json().get("status") == "success"
    assert "new_balance" in response.json()
