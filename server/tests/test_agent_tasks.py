from fastapi.testclient import TestClient
from server.mcp_server import app


def test_token_hashing():
    client = TestClient(app)
    resp = client.post("/auth/token")
    assert "token" in resp.json()


def test_wallet_functionality():
    client = TestClient(app)
    resp = client.post("/wallet/export", json={"user_id": "test"})
    assert "vials" in resp.json()


def test_api_pipeline():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_token_merge():
    client = TestClient(app)
    resp1 = client.post("/auth/token")
    assert len(resp1.json()["token"]) > 0
