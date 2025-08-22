from fastapi.testclient import TestClient
from server.mcp_server import app


def test_load_training():
    client = TestClient(app)
    for _ in range(100):
        resp = client.post("/agent/train", json={"vial_id": "all"})
        assert resp.status_code == 200
