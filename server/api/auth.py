from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")


@router.post("/token")
async def generate_token(network_id: str, session_id: str, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        token = f"token_{uuid.uuid4()}"
        await memory_manager.save_session(token, {"network_id": network_id, "session_id": session_id}, request_id)
        logger.info(f"Generated token for network_id {network_id}", request_id=request_id)
        return {"token": token, "request_id": request_id}
    except Exception as e:
        logger.error(f"Token generation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate")
async def validate_token(token: str = Depends(oauth2_scheme), memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        session = await memory_manager.get_session(token, request_id)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid token")
        logger.info(f"Validated token {token}", request_id=request_id)
        return {"status": "valid", "request_id": request_id}
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
