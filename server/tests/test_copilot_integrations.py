from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import pytest
import uuid


@pytest.fixture
def client():
    return TestClient(app)


def test_copilot_generate_config(client):
    request_id = str(uuid.uuid4())
    try:
        response = client.post(
            "/alchemist/vial/config",
            json={"prompt": "Generate FastAPI endpoint"},
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        assert "config" in response.json()
        logger.log("Copilot config generation test passed", request_id=request_id)
    except Exception as e:
        logger.log(f"Copilot test error: {str(e)}", request_id=request_id)
        raise


def test_copilot_deploy_vercel(client):
    request_id = str(uuid.uuid4())
    try:
        response = client.post(
            "/alchemist/deploy/vercel",
            json={"config": {"components": [], "connections": []}},
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        assert response.json().get("status") == "deployed"
        logger.log("Copilot Vercel deployment test passed", request_id=request_id)
    except Exception as e:
        logger.log(f"Vercel test error: {str(e)}", request_id=request_id)
        raise
