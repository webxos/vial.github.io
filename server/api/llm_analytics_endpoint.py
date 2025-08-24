from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.services.llm_router import get_llm_metrics
from server.config.database import get_db
from sqlalchemy.orm import Session
from fastapi.security import OAuth2AuthorizationCodeBearer
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO, filename="logs/llm_analytics.log")

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="/mcp/auth/authorize",
    tokenUrl="/mcp/auth/token",
    scheme_name="OAuth2PKCE"
)

class LLMAnalyticsRequest(BaseModel):
    provider: str = None  # Optional: filter by provider (e.g., "anthropic")
    time_range: str = "1h"  # e.g., "1h", "24h", "7d"

@router.get("/metrics")
async def get_llm_analytics(request: LLMAnalyticsRequest = Depends(), token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        # Prompt Shield validation
        if request.provider and any(word in request.provider.lower() for word in ["malicious", "hack"]):
            raise HTTPException(status_code=400, detail="Invalid provider name")
        
        metrics = await get_llm_metrics(db, provider=request.provider, time_range=request.time_range)
        return {
            "provider": request.provider or "all",
            "time_range": request.time_range,
            "metrics": {
                "average_latency_ms": metrics["avg_latency"],
                "error_rate": metrics["error_rate"],
                "request_count": metrics["request_count"]
            }
        }
    except Exception as e:
        logging.error(f"LLM analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
