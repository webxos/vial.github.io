from fastapi import APIRouter, Depends
from server.services.mcp_alchemist import Alchemist
from server.logging_config import logger
from fastapi.security import OAuth2PasswordBearer
import uuid

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.get("/troubleshoot/status")
async def troubleshoot_status(token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        alchemist = Alchemist()
        db_status = await alchemist.check_db_connection()
        agent_status = await alchemist.check_agent_availability()
        wallet_status = await alchemist.check_wallet_system()
        logger.info(
            f"Troubleshoot: DB={db_status}, Agents={agent_status}, Wallet={wallet_status}",
            request_id=request_id
        )
        return {
            "status": "healthy",
            "db": db_status,
            "agents": agent_status,
            "wallet": wallet_status,
            "request_id": request_id
        }
    except Exception as e:
        logger.error(f"Troubleshoot error: {str(e)}", request_id=request_id)
        return {"status": "unhealthy", "detail": str(e), "request_id": request_id}
