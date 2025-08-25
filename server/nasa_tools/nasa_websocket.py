from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter
from server.api.auth_endpoint import verify_token
import asyncio
import json
import websockets
import os

router = APIRouter(prefix="/masa/websocket", tags=["websocket"])

class NASAWebSocket:
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        uri = f"wss://nasa-websocket-endpoint.example.com/events"  # Placeholder URI
        async with websockets.connect(uri) as ws:
            try:
                while True:
                    message = await ws.recv()
                    await websocket.send_text(message)
            except websockets.ConnectionClosed:
                await websocket.close()

nasa_ws = NASAWebSocket()

@router.websocket("/events")
async def websocket_endpoint(websocket: WebSocket, token: dict = Depends(verify_token)):
    await nasa_ws.connect(websocket)
