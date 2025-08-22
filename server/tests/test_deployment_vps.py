from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_deploy_config(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    config_response = client.post("/save-config", json={
        "name": "test_config",
        "components": [{"id": "comp1", "type": "api_endpoint"}],
        "connections": []
    }, headers={"Authorization": f"Bearer {token}"})
    config_id = config_response.json()["config_id"]
    
    deploy_response = client.post("/deploy-config", json={"config_id": config_id}, headers={"Authorization": f"Bearer {token}"})
    assert deploy_response.status_code == 200
    assert deploy_response.json()["status"] == "deployed"
    assert "url" in deploy_response.json()
