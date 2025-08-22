import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_troubleshoot_endpoint(client):
    response = client.get("/troubleshoot")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert "checks" in response.json()

def test_troubleshoot_button():
    # Simulate button click and fetch response
    import requests
    response = requests.get("http://localhost:8000/troubleshoot")
    assert response.status_code == 200
    assert "running" in response.text
