from fastapi.testclient import TestClient
from ..main import app

client = TestClient(app)

def test_entangle_qubits():
    response = client.post("/api/auth/token", json={"username": "admin", "password": "admin"})
    token = response.json()["access_token"]
    response = client.post("/api/quantum/entangle", json={"link_id": "test-link", "qubit_count": 2}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "status" in response.json() and response.json()["status"] == "entangled"
