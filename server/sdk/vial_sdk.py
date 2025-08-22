from server.mcp_server import app
from fastapi.testclient import TestClient

def get_sdk_client():
    return TestClient(app)

def sync_quantum_state(vial_id: str):
    client = get_sdk_client()
    response = client.post("/quantum/sync", json={"vial_id": vial_id})
    return response.json()
