from fastapi.testclient import TestClient
from ..mcp_server import app
from ..services.vial_manager import VialManager
from ..models.visual_components import VisualConfig, ComponentModel, Position3D, ComponentType


def test_alchemist_auth():
    client = TestClient(app)
    response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_alchemist_train_with_config():
    client = TestClient(app)
    token_response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    token = token_response.json()["access_token"]
    config = VisualConfig(
        components=[
            ComponentModel(
                id="comp1",
                type=ComponentType.AGENT,
                title="Agent 1",
                position=Position3D(x=0, y=0, z=0),
                config={"vial_id": "vial1"},
                connections=[],
                svg_style="alert"
            )
        ],
        connections=[]
    )
    response = client.post("/alchemist/train", json={"prompt": "Generate agent config", "config": config.dict()}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "trained"
    assert "quantum_hash" in response.json()


def test_alchemist_wallet_export_with_svg():
    client = TestClient(app)
    token_response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    token = token_response.json()["access_token"]
    response = client.post("/alchemist/wallet/export", json={"user_id": "test", "svg_style": "alert"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "vials" in response.json()
    assert all("svg_diagram" in vial and "alert" in vial["svg_diagram"] for vial in response.json()["vials"])


def test_alchemist_troubleshoot():
    client = TestClient(app)
    token_response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    token = token_response.json()["access_token"]
    response = client.post("/alchemist/troubleshoot", json={"error": "Database connection failed"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "troubleshooting"
    assert "options" in response.json()


def test_alchemist_git_command():
    client = TestClient(app)
    token_response = client.post("/alchemist/auth/token", data={"username": "test", "password": "test"})
    token = token_response.json()["access_token"]
    response = client.post("/alchemist/git", json={"command": "status"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "executed"
