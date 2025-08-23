from fastapi import APIRouter, WebSocket, Depends
from server.api.mcp_tools import MCPTools
from server.logging import logger
from server.mcp.auth import oauth2_scheme
import json
import uuid


router = APIRouter()


@router.websocket("/mcp/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Depends(oauth2_scheme),
):
    request_id = str(uuid.uuid4())
    try:
        await websocket.accept()
        from server.mcp.auth import map_oauth_to_mcp_session

        await map_oauth_to_mcp_session(token, request_id)

        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            tool_name = request.get("tool")
            params = request.get("params", {})
            result = await MCPTools.execute_tool(tool_name, params)
            await websocket.send_text(
                json.dumps({
                    "status": "success",
                    "result": result,
                    "request_id": request_id,
                })
            )
            logger.log(
                f"WebSocket MCP tool executed: {tool_name}",
                request_id=request_id,
            )
    except Exception as e:
        logger.log(f"WebSocket error: {str(e)}", request_id=request_id)
        await websocket.send_text(
            json.dumps({
                "status": "error",
                "detail": str(e),
                "request_id": request_id,
            })
        )
        await websocket.close()
