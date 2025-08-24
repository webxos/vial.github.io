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

class PublishRequest(BaseModel):
    package_name: str
    version: str
    metadata: dict

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def publish_to_registry(package_name: str, version: str, metadata: dict) -> dict:
    """Publish package to MCP registry."""
    async with AsyncClient() as client:
        response = await client.post(
            "https://registry.modelcontextprotocol.org/publish",
            json={"name": package_name, "version": version, "metadata": metadata},
            headers={"Authorization": f"Bearer {os.getenv('REGISTRY_TOKEN')}"}
        )
        response.raise_for_status()
        return response.json()

@router.post("/mcp/registry/publish")
async def publish_package(request: PublishRequest, token: str = Security(oauth2_scheme)):
    """Publish package to MCP registry via API."""
    try:
        result = await publish_to_registry(request.package_name, request.version, request.metadata)
        logger.info(f"Published package {request.package_name} v{request.version}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Registry publishing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
