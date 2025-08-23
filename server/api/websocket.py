from fastapi import APIRouter, WebSocket, Depends
from server.services.mcp_alchemist import Alchemist
from server.logging_config import logger
from fastapi.security import OAuth2PasswordBearer
import json
import uuid
import asyncio

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.websocket("/mcp/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    await websocket.accept()
    try:
        alchemist = Alchemist()
        sessions = {}
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            task = message.get("task")
            params = message.get("params", {})
            session_id = str(uuid.uuid4())
            sessions[session_id] = {"task": task, "params": params}
            result = await alchemist.delegate_task(task, params)
            await websocket.send_text(json.dumps({
                "result": result,
                "request_id": request_id,
                "session_id": session_id
            }))
            logger.info(f"WebSocket task {task} processed", request_id=request_id)
            await asyncio.sleep(0.1)  # Prevent tight loop
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", request_id=request_id)
        await websocket.send_text(json.dumps({"error": str(e), "request_id": request_id}))
        await websocket.close()
    finally:
        sessions.clear()
