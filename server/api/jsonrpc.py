from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from server.services.advanced_logging import AdvancedLogger


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict
    id: str


router = APIRouter()
logger = AdvancedLogger()


@router.post("/jsonrpc")
async def jsonrpc_endpoint(request: JsonRpcRequest):
    if request.jsonrpc != "2.0":
        logger.log("Invalid JSON-RPC version", extra={"version": request.jsonrpc})
        raise HTTPException(status_code=400, detail="Invalid JSON-RPC version")
    
    methods = {
        "get_status": lambda params: {"status": "running", "version": "2.9.3"},
        "ping": lambda params: {"response": "pong"}
    }
    
    if request.method not in methods:
        logger.log("Unknown JSON-RPC method", extra={"method": request.method})
        raise HTTPException(status_code=400, detail="Method not found")
    
    result = methods[request.method](request.params)
    logger.log("JSON-RPC request processed", extra={"method": request.method, "id": request.id})
    return {"jsonrpc": "2.0", "result": result, "id": request.id}
