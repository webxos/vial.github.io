from fastapi import APIRouter, WebSocket
from server.security import verify_token

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()
    if not verify_token(token):
        await websocket.close(code=1008)
        return
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Echo: {data}")
