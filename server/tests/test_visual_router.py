from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_export_diagram(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    config_response = client.post("/save-config", json={
        "name": "test_config",
        "components": [{"id": "comp1", "type": "api_endpoint"}],
        "connections": []
    }, headers={"Authorization": f"Bearer {token}"})
    config_id = config_response.json()["config_id"]
    
    response = client.get(f"/visual/diagram/export?config_id={config_id}", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "exported"
    assert "svg_data" in response.json()
