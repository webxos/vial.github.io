from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


def test_terminal_commands(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    response = client.get("/commands", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "commands" in response.json()
    assert "help" in response.json()["commands"]
    assert response.json()["commands"]["help"] == "Display available commands"
