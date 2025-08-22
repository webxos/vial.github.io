from fastapi import APIRouter
from server.logging import logger

router = APIRouter()

@router.post("/jsonrpc")
async def jsonrpc_endpoint(data: dict):
    logger.log(f"JSON-RPC request: {data}")
    if data.get("method") == "ping":
        return {"result": "pong", "id": data.get("id")}
    return {"error": "Method not found", "id": data.get("id")}
