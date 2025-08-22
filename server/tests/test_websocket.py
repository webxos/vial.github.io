from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest
import json


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_websocket_collaboration(client: TestClient):
    async with client.websocket_connect("/ws") as websocket:
        # Test cursor update
        cursor_data = {"type": "cursor_update", "user_id": "test_user", "position": {"x": 100, "y": 200}}
        await websocket.send_text(json.dumps(cursor_data))
        response = await websocket.receive_json()
        assert response["type"] == "cursor_broadcast"
        assert response["user_id"] == "test_user"
        assert response["position"] == {"x": 100, "y": 200}
        
        # Test config update
        config_data = {"type": "config_update", "config": {"id": "config1", "name": "test_config"}}
        await websocket.send_text(json.dumps(config_data))
        response = await websocket.receive_json()
        assert response["type"] == "config_broadcast"
        assert response["config"]["id"] == "config1"
