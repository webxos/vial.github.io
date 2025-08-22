from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_prompt_training(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    token = token_response.json()["access_token"]
    response = client.post("/train-prompt", json={"prompt": "test prompt"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "trained"
    assert "output" in response.json()
