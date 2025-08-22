import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_api_access_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_api_access_troubleshoot(client):
    response = client.get("/troubleshoot")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
