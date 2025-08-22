from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_full_workflow(client: TestClient):
    # Authentication
    auth_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert auth_response.status_code == 200
    token = auth_response.json()["access_token"]

    # Train agents
    train_response = client.post("/agent/train", json={"vial_id": "vial1"}, headers={"Authorization": f"Bearer {token}"})
    assert train_response.json()["status"] == "trained"

    # Save config
    config_response = client.post("/save-config", json={"name": "test_config", "components": [{"id": "comp1", "type": "api_endpoint"}], "connections": []}, headers={"Authorization": f"Bearer {token}"})
    assert config_response.json()["status"] == "saved"

    # Export wallet
    wallet_response = client.post("/wallet/export", json={"user_id": "test_user"}, headers={"Authorization": f"Bearer {token}"})
    assert "vials" in wallet_response.json()

    # Quantum sync
    quantum_response = client.post("/quantum/sync", json={"vial_id": "vial1"}, headers={"Authorization": f"Bearer {token}"})
    assert "quantum_state" in quantum_response.json()
