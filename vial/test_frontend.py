import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.logging_config import logger
import uuid

@pytest.fixture
def client():
    return TestClient(app)

def test_frontend_wallet_render(client):
    request_id = str(uuid.uuid4())
    response = client.post(
        "/v1/wallet/export",
        json={"network_id": "54965687-3871-4f3d-a803-ac9840af87c4"},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "vial1, vial2, vial3, vial4" in response.json()["markdown"]
    logger.info("Frontend wallet render test passed", request_id=request_id)

def test_frontend_status_endpoint(client):
    request_id = str(uuid.uuid4())
    response = client.get("/v1/status")
    assert response.status_code == 200
    assert response.json()["status"] == "active"
    assert response.json()["version"] == "1.0.0"
    logger.info("Frontend status endpoint test passed", request_id=request_id)
