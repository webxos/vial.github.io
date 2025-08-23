from fastapi import APIRouter
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1/comms", tags=["comms"])


@router.post("/send_message")
async def send_message(message: dict, request_id: str = str(uuid.uuid4())):
    try:
        logger.info(f"Sending message: {message}", request_id=request_id)
        return {"status": "sent", "message": message, "request_id": request_id}
    except Exception as e:
        logger.error(f"Message send error: {str(e)}", request_id=request_id)
        raise
