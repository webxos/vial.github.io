```python
from fastapi import APIRouter, Security, HTTPException
from pydantic import BaseModel
from typing import Dict
import logging
import httpx
from server.config.settings import settings
from server.utils.security_sanitizer import sanitize_input

logger = logging.getLogger(__name__)
router = APIRouter()

class ServiceNowRequest(BaseModel):
    short_description: str
    description: str
    urgency: str = "low"

@router.post("/mcp/servicenow/ticket")
async def create_servicenow_ticket(request: ServiceNowRequest, token: str = Security(...)):
    """Create a ServiceNow ticket."""
    try:
        sanitized_request = ServiceNowRequest(
            short_description=sanitize_input(request.short_description),
            description=sanitize_input(request.description),
            urgency=sanitize_input(request.urgency)
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.SERVICENOW_INSTANCE}/api/now/table/incident",
                auth=(settings.SERVICENOW_USER, settings.SERVICENOW_PASSWORD),
                json=sanitized_request.dict(),
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"ServiceNow ticket created: {sanitized_request.short_description}")
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"ServiceNow request failed: {str(e)}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"ServiceNow endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```
