from fastapi import APIRouter, WebSocket, Depends
from server.services.mcp_alchemist import Alchemist
from server.mcp.auth import oauth2_scheme
from server.logging import logger
import uuid

router = APIRouter()

@router.websocket("/mcp/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        await websocket.accept()
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        alchemist = Alchemist()
        while True:
            data = await websocket.receive_json()
            task = data.get("tool")
            params = data.get("params", {})
            context = {"params": params}
            result = await alchemist.delegate_task(task, context)
            await websocket.send_json({"result": result, "request_id": request_id})
            logger.log(f"WebSocket task processed: {task}", request_id=request_id)
    except Exception as e:
        logger.log(f"WebSocket error: {str(e)}", request_id=request_id)
        await websocket.send_json({"error": str(e), "request_id": request_id})
        await websocket.close()
