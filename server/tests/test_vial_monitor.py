from fastapi.testclient import TestClient
from server.mcp_server import app

client = TestClient(app)


def test_monitor_endpoint():
    response = client.get("/monitor", headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 200
    assert "metrics" in response.json()


def test_monitor_unauthorized():
    response = client.get("/monitor")
    assert response.status_code == 401
    assert "Unauthorized" in response.json()["detail"]
