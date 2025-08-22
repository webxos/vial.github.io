from fastapi.testclient import TestClient
from ..mcp_server import app
from ..services.database import get_db
from ..models.webxos_wallet import WalletModel


def test_full_workflow():
    client = TestClient(app)
    # Authentication
    auth_response = client.post("/auth/token", data={"username": "test", "password": "test"})
    assert auth_response.status_code == 200
    token = auth_response.json()["access_token"]

    # Train agents
    train_response = client.post("/agent/train", json={"vial_id": "all"}, headers={"Authorization": f"Bearer {token}"})
    assert train_response.status_code == 200
    assert train_response.json()["status"] == "training_complete"

    # Export wallet
    export_response = client.post("/wallet/export", json={"user_id": "test"}, headers={"Authorization": f"Bearer {token}"})
    assert export_response.status_code == 200
    assert "vials" in export_response.json()

    # Quantum sync
    quantum_response = client.post("/quantum/sync", json={"vial_id": "vial1"}, headers={"Authorization": f"Bearer {token}"})
    assert quantum_response.status_code == 200
    assert "quantum_state" in quantum_response.json()
