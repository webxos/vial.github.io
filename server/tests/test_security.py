from fastapi.testclient import TestClient
from server.mcp_server import app

# ... (previous content assumed)

def test_security_headers():
    client = TestClient(app)
    resp = client.get("/health")
    assert "X-Content-Type-Options" in resp.headers

def test_oauth():
    client = TestClient(app)
    resp = client.post("/auth/token")
    assert resp.status_code == 200
