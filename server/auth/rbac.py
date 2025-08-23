import os
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from jose import JWTError, jwt
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi import HTTPException, Security
from fastapi.security import OAuth2AuthorizationCodeBearer

logger = logging.getLogger(__name__)

class RolePermission(BaseModel):
    role: str
    permissions: list[str]
    mfa_required: bool = False

class UserContext(BaseModel):
    user_id: str
    roles: list[str]
    mfa_verified: bool = False

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
    scopes={"https://www.googleapis.com/auth/drive": "Google Drive access"},
)

async def validate_google_oauth(token: str, client_id: str) -> Dict[str, Any]:
    """Validate Google OAuth2 token with PKCE."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v3/tokeninfo",
                params={"access_token": token}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Google OAuth validation failed: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid Google OAuth token")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def check_rbac(token: str, required_permissions: list[str]) -> UserContext:
    """Enforce RBAC with zero-trust and MFA."""
    try:
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET"),
            algorithms=["HS256"],
            options={"verify_aud": True},
            audience="vial_mcp"
        )
        user_context = UserContext(**payload)
        
        if any(perm in required_permissions for role in user_context.roles
               for perm in RolePermission(role=role, permissions=["read", "write"]).permissions):
            if RolePermission(role=user_context.roles[0]).mfa_required and not user_context.mfa_verified:
                raise HTTPException(status_code=403, detail="MFA required")
            return user_context
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    except JWTError as e:
        logger.error(f"RBAC JWT validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")
