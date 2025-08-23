from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid

router = APIRouter()

class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict
    id: str

@router.post("/jsonrpc")
async def jsonrpc_endpoint(request: JsonRpcRequest):
    request_id = str(uuid.uuid4())
    try:
        if request.jsonrpc != "2.0":
            raise HTTPException(status_code=400, detail="Invalid JSON-RPC version")
        alchemist = Alchemist()
        result = await alchemist.delegate_task(request.method, request.params)
        logger.info(f"JSON-RPC method {request.method} executed", request_id=request_id)
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request.id,
            "request_id": request_id
        }
    except Exception as e:
        logger.error(f"JSON-RPC error: {str(e)}", request_id=request_id)
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": str(e)},
            "id": request.id,
            "request_id": request_id
        }
