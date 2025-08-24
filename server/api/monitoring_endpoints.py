```python
from fastapi import APIRouter, Security, HTTPException
from typing import Dict
import logging
import psutil
from server.config.settings import settings
from server.utils.security_sanitizer import sanitize_input

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/mcp/monitoring/health")
async def health_check(token: str = Security(...)) -> Dict:
    """Check server health and resource usage."""
    try:
        # Check CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Verify critical services
        services = {
            "llm_router": "healthy" if settings.ANTHROPIC_API_KEY or settings.MISTRAL_API_KEY else "unconfigured",
            "obs": "healthy" if settings.OBS_HOST and settings.OBS_PORT else "unconfigured",
            "servicenow": "healthy" if settings.SERVICENOW_INSTANCE else "unconfigured"
        }

        health_status = {
            "status": "healthy",
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory.percent,
            "disk_usage_percent": disk.percent,
            "services": services
        }
        logger.info(f"Health check: {health_status}")
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```
