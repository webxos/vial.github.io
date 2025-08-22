from fastapi.testclient import TestClient
from server.mcp_server import app

# ... (previous content assumed)

def test_copilot_integration():
    client = TestClient(app)
    resp = client.post("/copilot/integrate")
    assert resp.status_code == 200

def test_copilot_data():
    client = TestClient(app)
    resp = client.get("/copilot/data")
    assert "data" in resp.json()
