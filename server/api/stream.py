from fastapi import WebSocket, WebSocketDisconnect
from server.mcp_server import app


async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            await websocket.send_json({"vial_id": data.get("vial_id"), "state": "active"})
    except WebSocketDisconnect:
        pass
