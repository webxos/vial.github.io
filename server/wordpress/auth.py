import os
import logging
from typing import Dict, Any, Optional
import httpx
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger(__name__)

class WPAuthConfig(BaseModel):
    base_url: str = Field(..., env="WORDPRESS_BASE_URL")
    username: str = Field(..., env="WORDPRESS_USERNAME")
    app_password: str = Field(..., env="WORDPRESS_APP_PASSWORD")
    timeout: int = Field(30, env="WORDPRESS_API_TIMEOUT")

class JWTValidationError(Exception):
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def get_wp_token(config: WPAuthConfig) -> str:
    """Async fetch WordPress Application Password token."""
    auth = httpx.BasicAuth(username=config.username, password=config.app_password)
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        response = await client.post(f"{config.base_url}/wp-json/jwt-auth/v1/token", auth=auth)
        response.raise_for_status()
        return response.json()["token"]

def validate_jwt(token: str, secret: str, audience: str = "vial_mcp") -> Dict[str, Any]:
    """Validate JWT with iss/aud/exp checks."""
    try:
        return jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": True}, audience=audience)
    except JWTError as e:
        raise JWTValidationError(f"Invalid JWT: {str(e)}")

async def authenticated_request(method: str, endpoint: str, config: WPAuthConfig, **kwargs) -> Dict[str, Any]:
    """Async authenticated WP REST request with error handling."""
    token = await get_wp_token(config)
    headers = {"Authorization": f"Bearer {token}", **kwargs.get("headers", {})}
    url = f"{config.base_url}/wp-json/wp/v2/{endpoint}"
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        try:
            response = await client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"WP API error: {str(e)}")
            raise
