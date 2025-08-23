from fastapi import APIRouter
from server.api import auth, jsonrpc, monitoring, troubleshoot, upload, websocket
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1", tags=["v1"])

@router.get("/status")
async def api_status():
    request_id = str(uuid.uuid4())
    logger.info("API v1 status checked", request_id=request_id)
    return {"status": "active", "version": "1.0.0", "request_id": request_id}

router.include_router(auth.router, prefix="/auth")
router.include_router(jsonrpc.router, prefix="/jsonrpc")
router.include_router(monitoring.router, prefix="/monitoring")
router.include_router(troubleshoot.router, prefix="/troubleshoot")
router.include_router(upload.router, prefix="/upload")
router.include_router(websocket.router, prefix="/mcp")
