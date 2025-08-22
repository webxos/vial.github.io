from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest
import json


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_comms_hub(client: TestClient):
    async with client.websocket_connect("/comms") as websocket:
        message_data = {"type": "message", "user_id": "test_user", "message": "Hello, team!"}
        await websocket.send_text(json.dumps(message_data))
        response = await websocket.receive_json()
        assert response["type"] == "message_broadcast"
        assert response["user_id"] == "test_user"
        assert response["message"] == "Hello, team!"
