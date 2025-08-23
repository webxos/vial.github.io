import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
import asyncio

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_load_vial_status():
    async def make_request(client):
        response = await client.post(
            "/mcp/vial_status_get",
            json={"vial_id": "test_vial"},
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        return response.json()

    tasks = [make_request(TestClient(app)) for _ in range(10)]
    results = await asyncio.gather(*tasks)
    for result in results:
        assert result.get("vial_id") == "test_vial"
        assert "balance" in result

@pytest.mark.asyncio
async def test_load_quantum_circuit():
    async def make_request(client):
        response = await client.post(
            "/quantum/circuit",
            json={"qubits": 2, "gates": ["h", "cx"]},
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        return response.json()

    tasks = [make_request(TestClient(app)) for _ in range(10)]
    results = await asyncio.gather(*tasks)
    for result in results:
        assert "circuit" in result
