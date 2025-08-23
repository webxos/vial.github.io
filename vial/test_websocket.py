import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.logging_config import logger
import json
import uuid
import websockets

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async async def test_websocket_task_execution():
    request_id = str(uuid.uuid4())
    async with websockets.connect("ws://localhost:8000/v1/mcp/ws?token=test_token") as websocket:
        await websocket.send(json.dumps({
            "task_name": "train_model",
            "params": {"vial_id": "vial1", "network_id": "54965687-3871-4f3d-a803-ac9840af87c4"},
            "session_token": "test_token"
        }))
        response = await websocket.recv()
        data = json.loads(response)
        assert data["result"]["status"] == "trained"
        assert data["request_id"] == request_id
        logger.info("WebSocket task execution test passed", request_id=request_id)

@pytest.mark.asyncio
async async def test_websocket_session_save():
    request_id = str(uuid.uuid4())
    async with websockets.connect("ws://localhost:8000/v1/mcp/ws?token=test_token") as websocket:
        await websocket.send(json.dumps({
            "task_name": "create_agent",
            "params": {
                "vial_id": "vial5",
                "menu_info": {"selectedTool": "agent", "guideStep": 1},
                "build_progress": ["vial5"]
            },
            "session_token": "test_token"
        }))
        response = await websocket.recv()
        data = json.loads(response)
        assert data["result"]["status"] == "success"
        assert data["request_id"] == request_id
        logger.info("WebSocket session save test passed", request_id=request_id)
