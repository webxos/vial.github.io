from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from httpx import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from fastapi.security import OAuth2AuthorizationCodeBearer
from server.config.settings import settings
from server.utils.security_sanitizer import sanitize_input

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

class DjangoAdminRequest(BaseModel):
    resource: str
    action: str
    data: dict

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_django_admin(resource: str, action: str, data: dict) -> dict:
    """Call Django admin interface via REST API."""
    try:
        sanitized_data = sanitize_input(data)
        async with AsyncClient() as client:
            response = await client.post(
                f"{settings.DJANGO_ADMIN_URL}/{resource}/{action}",
                json=sanitized_data,
                headers={"Authorization": f"Bearer {settings.DJANGO_API_KEY}"}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Django admin call failed: {str(e)}")
        raise

@router.post("/mcp/django/admin")
async def django_admin(request: DjangoAdminRequest, token: str = Security(oauth2_scheme)):
    """Access Django admin interface for quantum and RAG operations."""
    try:
        result = await call_django_admin(request.resource, request.action, request.data)
        logger.info(f"Django admin accessed: {request.resource}/{request.action}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Django admin request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
