import logging
import subprocess
from pydantic import BaseModel
from httpx import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class DeploymentStatus(BaseModel):
    status: str
    health: bool

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def validate_deployment():
    """Validate MCP server deployment."""
    try:
        # Check Kubernetes deployment status
        result = subprocess.run(
            ["kubectl", "rollout", "status", "deployment/vial-mcp", "--timeout=5m"],
            capture_output=True, text=True, check=True
        )
        status = "deployed" if "successfully rolled out" in result.stdout else "failed"

        # Check health endpoint
        async with AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            response.raise_for_status()
            health = response.json().get("status") == "healthy"

        # Placeholder: OBS/SVG health check
        # response = await client.get("http://localhost:8000/obs/health")
        # obs_health = response.json().get("status") == "healthy"

        return DeploymentStatus(status=status, health=health)
    except Exception as e:
        logger.error(f"Deployment validation failed: {str(e)}")
        raise
