from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_quantum_sync(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    response = client.post("/quantum/sync", json={"vial_id": "vial1"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "quantum_state" in response.json()
    assert response.json()["vial_id"] == "vial1"
