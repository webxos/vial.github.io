from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_generate_code(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    component = {
        "id": "comp1",
        "type": "api_endpoint",
        "title": "Test Endpoint",
        "position": {"x": 0, "y": 0},
        "config": {},
        "connections": []
    }
    response = client.post("/copilot/generate-code", json={"component": component}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "generated"
    assert "code" in response.json()
