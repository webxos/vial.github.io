from fastapi import APIRouter, HTTPException, Security
from prometheus_client import Counter, Gauge, generate_latest
from fastapi.responses import PlainTextResponse
from fastapi.security import OAuth2AuthorizationCodeBearer
from server.config.settings import settings
from server.utils.security_sanitizer import sanitize_input
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

# Prometheus metrics
request_counter = Counter("vial_mcp_requests_total", "Total requests to MCP endpoints", ["endpoint"])
system_health = Gauge("vial_mcp_system_health", "System health status", ["service"])

@router.get("/metrics")
async def get_metrics(token: str = Security(oauth2_scheme)):
    """Expose Prometheus metrics for MCP system."""
    try:
        # Update metrics (placeholder for system health)
        system_health.labels(service="quantum").set(1)
        system_health.labels(service="rag").set(1)
        system_health.labels(service="api").set(1)

        # Increment request counter
        request_counter.labels(endpoint="/metrics").inc()

        logger.info("Metrics endpoint accessed")
        return PlainTextResponse(generate_latest())
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
