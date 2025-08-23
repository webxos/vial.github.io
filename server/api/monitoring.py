from fastapi import APIRouter, Depends
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
from fastapi.security import OAuth2PasswordBearer
import uuid

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.get("/monitoring/health")
async def health_check(token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        alchemist = Alchemist()
        db_status = await alchemist.check_db_connection()
        agent_status = await alchemist.check_agent_availability()
        wallet_status = await alchemist.check_wallet_system()
        response_time = await alchemist.get_api_response_time()
        logger.info(
            f"Health check: DB={db_status}, Agents={agent_status}, Wallet={wallet_status}",
            request_id=request_id
        )
        return {
            "status": "healthy",
            "db": db_status,
            "agents": agent_status,
            "wallet": wallet_status,
            "response_time": response_time,
            "request_id": request_id
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", request_id=request_id)
        return {"status": "unhealthy", "detail": str(e), "request_id": request_id}

@router.get("/monitoring/logs")
async def get_logs(token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        with open("errorlog.md", "r") as f:
            logs = f.read()
        logger.info(f"Retrieved logs", request_id=request_id)
        return {"logs": logs, "request_id": request_id}
    except Exception as e:
        logger.error(f"Log retrieval error: {str(e)}", request_id=request_id)
        return {"status": "error", "detail": str(e), "request_id": request_id}
