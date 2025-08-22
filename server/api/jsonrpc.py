from fastapi import APIRouter
from server.services.advanced_logging import AdvancedLogger
import json


router = APIRouter()
logger = AdvancedLogger()


@router.post("/jsonrpc")
async def jsonrpc_endpoint(request: dict):
    method = request.get("method")
    params = request.get("params", {})
    response = {"jsonrpc": "2.0", "id": request.get("id")}
    if method == "ping":
        response["result"] = "pong"
    else:
        response["error"] = {"code": -32601, "message": "Method not found"}
    logger.log("JSON-RPC request processed",
               extra={"method": method})
    return response
