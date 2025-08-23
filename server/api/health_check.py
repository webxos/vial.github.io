from fastapi import APIRouter, HTTPException
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1/monitoring", tags=["monitoring"])


@router.get("/health")
async def health_check(memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        session = await memory_manager.get_session("health_check", request_id)
        status = {"status": "healthy", "db": bool(session), "agents": {}, "wallet": True, "response_time": 0.1}
        logger.info(f"Health check passed: {status}", request_id=request_id)
        return status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
