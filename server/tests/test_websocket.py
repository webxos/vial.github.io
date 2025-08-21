from fastapi.testclient import TestClient
from server.mcp_server import app
import websockets
import pytest


async def auth_token():
    client = TestClient(app)
    response = client.post("/auth/token", data={"username": "admin",
                                                "password": "admin"})
    return response.json()["access_token"]


@pytest.mark.asyncio
async def test_websocket_connection(auth_token):
    async with websockets.connect("ws://localhost:8000/ws?token=" + auth_token) as ws:
        await ws.send("test message")
        response = await ws.recv()
        assert response == "Echo: test message"


@pytest.mark.asyncio
async def test_websocket_invalid_token():
    with pytest.raises(websockets.exceptions.ConnectionClosed):
        async with websockets.connect("ws://localhost:8000/ws?token=invalid") as ws:
            await ws.send("test message")
