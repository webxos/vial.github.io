from typing import List, Dict
from litellm import completion
from server.models.analytics_repository import AnalyticsRepository
from server.config.database import get_db
from sqlalchemy.orm import Session
import time
import logging

logging.basicConfig(level=logging.INFO, filename="logs/llm_router.log")

async def route_to_llm(query: str, provider: str, db: Session = Depends(get_db)) -> List[str]:
    """Route query to specified LLM provider and log analytics."""
    try:
        start_time = time.time()
        analytics_repo = AnalyticsRepository(db)
        
        # Supported providers
        providers = ["anthropic", "mistral", "google", "xai", "meta", "local"]
        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Placeholder: Call LLM using litellm
        response = await completion(model=f"{provider}/default", messages=[{"role": "user", "content": query}])
        results = [msg["content"] for msg in response.choices]
        
        # Log analytics
        latency_ms = (time.time() - start_time) * 1000
        analytics_repo.log_request(provider=provider, latency_ms=latency_ms, success=True)
        
        return results
    except Exception as e:
        analytics_repo.log_request(provider=provider, latency_ms=(time.time() - start_time) * 1000, success=False)
        logging.error(f"LLM routing error for {provider}: {str(e)}")
        raise ValueError(f"LLM routing failed: {str(e)}")

async def get_llm_metrics(db: Session, provider: str = None, time_range: str = "1h") -> Dict:
    """Retrieve LLM performance metrics."""
    try:
        analytics_repo = AnalyticsRepository(db)
        return analytics_repo.get_metrics(provider=provider, time_range=time_range)
    except Exception as e:
        logging.error(f"LLM metrics error: {str(e)}")
        raise ValueError(f"Metrics retrieval failed: {str(e)}")
