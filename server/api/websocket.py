from fastapi import APIRouter, WebSocket, Depends
from server.security import verify_token
from server.logging import logger

router = APIRouter(tags=["websocket"])

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Depends(verify_token)):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"WebSocket received: {data}")
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()
