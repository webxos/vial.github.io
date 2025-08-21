import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


client = TestClient(app)


@pytest.mark.asyncio
async def test_quantum_execute():
    token_response = await client.post("/auth/token")
    token = token_response.json()["access_token"]
    circuit = {
        "qubits": 2,
        "gates": [{"type": "h", "target": 0}, {"type": "cx", "control": 0, "target": 1}]
    }
    response = await client.post(
        "/quantum/execute",
        json={"circuit": circuit, "backend": "qasm_simulator"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "success"
    assert "counts" in response.json()
