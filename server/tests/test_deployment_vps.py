from fastapi.testclient import TestClient
from server.mcp_server import app

# ... (previous content assumed)

def test_deployment():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
