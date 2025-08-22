from fastapi import APIRouter

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket):
    await websocket.accept()
    await websocket.send_text("Connected")
