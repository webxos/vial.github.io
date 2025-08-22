from fastapi.testclient import TestClient
from server.mcp_server import app
def test_integration():
    client = TestClient(app)
    resp = client.post("/quantum/sync", json={"vial_id": "vial1"})
    assert resp.status_code == 200
