from fastapi import APIRouter, WebSocket
from server.services.advanced_logging import AdvancedLogger
import json


router = APIRouter()
logger = AdvancedLogger()


@router.websocket("/comms")
async def comms_hub(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.log("Message received",
                       extra={"type": message.get("type"), "user_id": message.get("user_id")})
            await websocket.send_json({"type": "message_broadcast", "data": message})
    except Exception as e:
        logger.log("Comms hub error", extra={"error": str(e)})
        await websocket.close()
