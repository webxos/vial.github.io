import pytest
import websockets
import json
import asyncio
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.mcp_alchemist import Alchemist
from server.webxos_wallet import WebXOSWallet
from server.logging_config import logger

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def websocket_client():
    async with websockets.connect("ws://localhost:8000/mcp/ws") as ws:
        yield ws

@pytest.fixture
def wallet():
    return WebXOSWallet()

@pytest.mark.asyncio
async def test_websocket_session(client, websocket_client):
    request_id = "test-123"
    await websocket_client.send(json.dumps({
        "task": "vial_train",
        "params": {"vial_id": "vial1", "network_id": "54965687-3871-4f3d-a803-ac9840af87c4"}
    }))
    response = json.loads(await websocket_client.recv())
    assert response["request_id"] == request_id
    assert response["result"]["status"] == "trained"
    logger.info("WebSocket session test passed", request_id=request_id)

@pytest.mark.asyncio
async def test_wallet_import_export(client, wallet):
    network_id = "54965687-3871-4f3d-a803-ac9840af87c4"
    markdown = """
# Vial Wallet Export
**Timestamp**: 2025-08-23T01:35:00Z
**User ID**: 54965687-3871-4f3d-a803-ac9840af87c4
**Wallet Address**: 54965687-3871-4f3d-a803-ac9840af87c4
**Balance**: 764.0000 $WEBXOS
**Vials**: vial1, vial2, vial3, vial4
"""
    response = client.post(
        "/wallet/import",
        json={"file": markdown},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["network_id"] == network_id
    export_response = client.post(
        "/wallet/export",
        json={"network_id": network_id},
        headers={"Authorization": "Bearer test_token"}
    )
    assert export_response.status_code == 200
    assert network_id in export_response.json()["markdown"]
    logger.info("Wallet import/export test passed", request_id=str(uuid.uuid4()))

@pytest.mark.asyncio
async def test_agent_coordination(client):
    network_id = "54965687-3871-4f3d-a803-ac9840af87c4"
    response = client.post(
        "/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "agent_coord",
            "params": {"network_id": network_id},
            "id": "1"
        }
    )
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "coordinated"
    assert len(response.json()["result"]["results"]) == 4
    logger.info("Agent coordination test passed", request_id=str(uuid.uuid4()))
