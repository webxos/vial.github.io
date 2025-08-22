from fastapi import APIRouter, WebSocket
from server.services.advanced_logging import AdvancedLogger
import json


router = APIRouter()
logger = AdvancedLogger()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.log("WebSocket connection established", extra={"client": websocket.client})
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            if payload.get("type") == "cursor_update":
                await websocket.send_json({
                    "type": "cursor_broadcast",
                    "user_id": payload.get("user_id"),
                    "position": payload.get("position")
                })
                logger.log("Cursor update broadcast", extra={"user_id": payload.get("user_id")})
            elif payload.get("type") == "config_update":
                await websocket.send_json({
                    "type": "config_broadcast",
                    "config": payload.get("config")
                })
                logger.log("Config update broadcast", extra={"config_id": payload.get("config").get("id")})
    except Exception as e:
        logger.log("WebSocket error", extra={"error": str(e)})
        await websocket.close()
