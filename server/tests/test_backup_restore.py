from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_backup_restore(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    # Save config
    config_response = client.post("/save-config", json={
        "name": "test_config",
        "components": [{"id": "comp1", "type": "api_endpoint"}],
        "connections": []
    }, headers={"Authorization": f"Bearer {token}"})
    assert config_response.status_code == 200
    
    # Backup
    backup_response = client.post("/backup", headers={"Authorization": f"Bearer {token}"})
    assert backup_response.status_code == 200
    assert backup_response.json()["status"] == "backed up"
    
    # Restore
    restore_response = client.post("/restore", headers={"Authorization": f"Bearer {token}"})
    assert restore_response.status_code == 200
    assert restore_response.json()["status"] == "restored"
