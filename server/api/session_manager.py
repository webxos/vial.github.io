from fastapi import APIRouter, Depends, HTTPException
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1/session", tags=["session"])


@router.post("/cancel")
async def cancel_session(token: str, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        session = await memory_manager.get_session(token, request_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        await memory_manager.delete_session(token, request_id)
        logger.info(f"Cancelled session {token}", request_id=request_id)
        return {"status": "cancelled", "request_id": request_id}
    except Exception as e:
        logger.error(f"Session cancellation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{token}")
async def get_session_status(token: str, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        session = await memory_manager.get_session(token, request_id)
        status = "active" if session else "inactive"
        logger.info(f"Session status for {token}: {status}", request_id=request_id)
        return {"status": status, "request_id": request_id}
    except Exception as e:
        logger.error(f"Session status error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
