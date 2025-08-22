from fastapi import APIRouter, WebSocket
from server.logging import logger
from server.services.vial_manager import VialManager
from server.api.websocket import broadcast_message

router = APIRouter()


@router.websocket("/comms")
async def comms_hub(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            vial_id = data.get("vial_id")
            message = data.get("message")
            if vial_id and message:
                await broadcast_message(f"{vial_id}: {message}")
                logger.log(f"Comms hub sent: {vial_id} - {message}")
                await websocket.send_json({"status": "sent", "vial_id": vial_id})
            else:
                await websocket.send_json({"error": "Invalid data"})
    except Exception as e:
        logger.log(f"Comms hub error: {str(e)}")
        await websocket.close()
