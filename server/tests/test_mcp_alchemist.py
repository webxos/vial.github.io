import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.webxos_wallet import WalletModel
import uuid
import os
import json


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_db():
    class MockSession:
        def __init__(self):
            self.data = []
        def add(self, item):
            self.data.append(item)
        def commit(self):
            pass
        def query(self, model):
            return self
        def filter(self, *args):
            return self
        def all(self):
            return [WalletModel(id=str(uuid.uuid4()), user_id="test", address="test_addr", balance=100.0, hash="test_hash")]
    return MockSession()


def test_vial_status_get(client, mock_db):
    vial_id = str(uuid.uuid4())
    response = client.post(
        "/alchemist/vial/status",
        json={"vial_id": vial_id},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "status" in response.json()


def test_vial_config_generate(client):
    response = client.post(
        "/alchemist/vial/config",
        json={"prompt": "Generate config"},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "config" in response.json()


def test_deploy_vercel(client):
    config = {"components": [], "connections": []}
    config_path = f"resources/configs/test_{uuid.uuid4()}.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f)
    response = client.post(
        "/alchemist/deploy/vercel",
        json={"config_path": config_path},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json().get("status") == "deployed"


def test_git_commit_push(client):
    response = client.post(
        "/alchemist/git/commit",
        json={"message": "Test commit"},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json().get("status") == "committed"


def test_quantum_circuit_build(client):
    response = client.post(
        "/alchemist/quantum/circuit",
        json={"components": []},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "circuit" in response.json()
