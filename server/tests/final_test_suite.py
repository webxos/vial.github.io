from fastapi.testclient import TestClient
from server.mcp_server import app


def test_final():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
