from fastapi.testclient import TestClient
from server.mcp_server import app


def test_troubleshoot_endpoint():
    client = TestClient(app)
    resp = client.get("/troubleshoot")
    assert resp.status_code == 200


def test_error_logging():
    client = TestClient(app)
    resp = client.get("/troubleshoot")
    assert "error" not in resp.text


def test_system_check():
    client = TestClient(app)
    resp = client.get("/troubleshoot")
    assert "system" in resp.text
