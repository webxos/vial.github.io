from fastapi import APIRouter, WebSocket
from server.api.websocket import broadcast_message
from server.logging import logger
import uuid
import json

router = APIRouter()

@router.websocket("/comms/hub")
async def comms_hub_endpoint(websocket: WebSocket):
    request_id = str(uuid.uuid4())
    try:
        await websocket.accept()
        while True:
            data = await websocket.receive_json()
            task = data.get("task")
            params = data.get("params", {})
            result = await broadcast_message(task, params, request_id)
            await websocket.send_json({
                "status": "success",
                "result": result,
                "request_id": request_id
            })
            logger.log(f"Comms hub processed task: {task}", request_id=request_id)
    except Exception as e:
        logger.log(f"Comms hub error: {str(e)}", request_id=request_id)
        await websocket.send_json({
            "status": "error",
            "detail": str(e),
            "request_id": request_id
        })
        await websocket.close()
