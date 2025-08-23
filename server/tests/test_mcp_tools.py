import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.api.mcp_tools import MCPTools
from server.services.mcp_alchemist import Alchemist

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def alchemist():
    return Alchemist()

@pytest.mark.asyncio
async def test_vial_status_get(client, alchemist):
    vial_id = "test_vial"
    response = await client.post(
        "/mcp/vial_status_get",
        json={"vial_id": vial_id},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json().get("vial_id") == vial_id

@pytest.mark.asyncio
async def test_quantum_circuit_build(client, alchemist):
    response = await client.post(
        "/mcp/quantum_circuit_build",
        json={"qubits": 2, "gates": ["h", "cx"]},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "circuit" in response.json()

@pytest.mark.asyncio
async def test_git_commit_push(client, alchemist):
    response = await client.post(
        "/mcp/git_commit_push",
        json={"repo_path": ".", "commit_message": "Test commit"},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json().get("status") == "success"
