import pytest
import asyncio
import websockets
import json
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.logging_config import logger
import time
import uuid

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_websocket_load():
    request_id = str(uuid.uuid4())
    async def send_request(ws, i):
        await ws.send(json.dumps({
            "task": "vial_train",
            "params": {"vial_id": f"vial{i%4+1}", "network_id": "54965687-3871-4f3d-a803-ac9840af87c4"}
        }))
        start_time = time.time()
        response = json.loads(await ws.recv())
        latency = (time.time() - start_time) * 1000
        assert latency < 200
        assert response["result"]["status"] == "trained"
        return latency

    tasks = []
    async with websockets.connect("ws://localhost:8000/v1/mcp/ws") as ws:
        for i in range(100):
            tasks.append(send_request(ws, i))
        latencies = await asyncio.gather(*tasks)
    avg_latency = sum(latencies) / len(latencies)
    logger.info(f"WebSocket load test: avg latency {avg_latency:.2f}ms", request_id=request_id)
    assert avg_latency < 200

@pytest.mark.asyncio
async def test_agent_coordination_load(client):
    request_id = str(uuid.uuid4())
    network_id = "54965687-3871-4f3d-a803-ac9840af87c4"
    start_time = time.time()
    for _ in range(10):
        response = client.post(
            "/v1/jsonrpc",
            json={
                "jsonrpc": "2.0",
                "method": "agent_coord",
                "params": {"network_id": network_id},
                "id": str(uuid.uuid4())
            }
        )
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "coordinated"
        assert len(response.json()["result"]["results"]) == 4
    latency = (time.time() - start_time) * 1000 / 10
    logger.info(f"Agent coordination load test: avg latency {latency:.2f}ms", request_id=request_id)
    assert latency < 200
