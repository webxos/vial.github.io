from fastapi import APIRouter, WebSocket
from server.services.advanced_logging import AdvancedLogger
import json


router = APIRouter()
logger = AdvancedLogger()


@router.websocket("/comms")
async def comms_hub(websocket: WebSocket):
    await websocket.accept()
    logger.log("Comms hub connection established", extra={"client": websocket.client})
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            if payload.get("type") == "message":
                await websocket.send_json({
                    "type": "message_broadcast",
                    "user_id": payload.get("user_id"),
                    "message": payload.get("message")
                })
                logger.log("Message broadcast", extra={"user_id": payload.get("user_id")})
    except Exception as e:
        logger.log("Comms hub error", extra={"error": str(e)})
        await websocket.close()
