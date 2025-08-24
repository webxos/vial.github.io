from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.models.analytics_repository import AnalyticsRepository
from server.models.dao_repository import DAORepository
from server.config.database import get_db
from sqlalchemy.orm import Session
from fastapi.security import OAuth2AuthorizationCodeBearer
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO, filename="logs/monitoring.log")

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="/mcp/auth/authorize",
    tokenUrl="/mcp/auth/token",
    scheme_name="OAuth2PKCE"
)

class MetricsRequest(BaseModel):
    time_range: str = "1h"  # e.g., "1h", "24h", "7d"
    agent_type: str = None  # e.g., "alchemist", "swarm"

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/metrics")
async def get_system_metrics(request: MetricsRequest = Depends(), token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        analytics_repo = AnalyticsRepository(db)
        dao_repo = DAORepository(db)
        
        # Get LLM metrics
        llm_metrics = analytics_repo.get_metrics(time_range=request.time_range)
        
        # Get DAO reputation stats
        total_reputation = sum(
            dao_repo.get_reputation(wallet.wallet_id)
            for wallet in db.query(dao_repo.DAOReputation).all()
        )
        
        return {
            "llm_metrics": llm_metrics,
            "dao_total_reputation": total_reputation,
            "agent_type": request.agent_type or "all",
            "time_range": request.time_range
        }
    except Exception as e:
        logging.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
