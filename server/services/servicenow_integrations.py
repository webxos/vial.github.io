from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from httpx import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import logging
from fastapi.security import OAuth2AuthorizationCodeBearer

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

class ServiceNowRequest(BaseModel):
    endpoint: str
    params: dict

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def query_servicenow_api(endpoint: str, params: dict) -> dict:
    """Query ServiceNow API with given endpoint and parameters."""
    async with AsyncClient() as client:
        response = await client.get(
            f"https://{os.getenv('SERVICENOW_INSTANCE')}.service-now.com/api/now/{endpoint}",
            params=params,
            auth=(os.getenv("SERVICENOW_API_KEY"), "")
        )
        response.raise_for_status()
        return response.json()

@router.post("/mcp/tools/servicenow")
async def query_servicenow(request: ServiceNowRequest, token: str = Security(oauth2_scheme)):
    """Query ServiceNow API via MCP."""
    try:
        result = await query_servicenow_api(request.endpoint, request.params)
        logger.info(f"ServiceNow API queried: {request.endpoint}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"ServiceNow API query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
