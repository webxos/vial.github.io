from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_stream_logs(client: TestClient):
    token_response = client.post("/auth/token", data={"username": "admin", "password": "secret"})
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    
    with client.websocket_connect("/stream/logs") as websocket:
        data = websocket.receive_text()
        assert "data: Log event" in data
        assert "2025" in data
