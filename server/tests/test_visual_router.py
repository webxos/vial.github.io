from fastapi.testclient import TestClient
from ..mcp_server import app
from ..models.visual_components import ComponentType


def test_visual_config_creation():
    client = TestClient(app)
    config = {
        "components": [
            {
                "id": "comp1",
                "type": ComponentType.API_ENDPOINT,
                "title": "Test API",
                "position": {"x": 0, "y": 0, "z": 0},
                "config": {},
                "connections": []
            }
        ],
        "connections": []
    }
    response = client.post("/save-config", json=config)
    assert response.status_code == 200
    assert response.json()["status"] == "saved"


def test_component_validation():
    client = TestClient(app)
    component = {
        "id": "comp1",
        "type": ComponentType.API_ENDPOINT,
        "title": "Test API",
        "position": {"x": 0, "y": 0, "z": 0},
        "config": {},
        "connections": []
    }
    response = client.post("/components/validate", json=component)
    assert response.status_code == 200
    assert response.json()["status"] == "valid"


def test_deployment_pipeline():
    client = TestClient(app)
    response = client.get("/components/available")
    assert response.status_code == 200
    assert len(response.json()["components"]) >= 5
