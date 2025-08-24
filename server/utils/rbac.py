from fastapi import Security, HTTPException
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel
from typing import List, Dict
import logging
from server.config.settings import settings

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

class RoleCheck(BaseModel):
    roles: List[str]
    endpoint: str

class RBAC:
    def __init__(self):
        self.role_permissions = {
            "admin": ["/mcp/tools/*", "/mcp/quantum", "/mcp/nasa", "/mcp/plugins"],
            "user": ["/mcp/tools/servicenow", "/mcp/tools/alibaba", "/mcp/llm"],
            "quantum_user": ["/mcp/quantum"],
            "rag_user": ["/mcp/rag"]
        }

    def check_access(self, user_roles: List[str], endpoint: str) -> bool:
        """Check if user has access to the endpoint based on roles."""
        for role in user_roles:
            if role in self.role_permissions:
                for allowed_endpoint in self.role_permissions[role]:
                    if endpoint.startswith(allowed_endpoint.replace("*", "")):
                        return True
        return False

rbac = RBAC()

async def get_current_user_roles(token: str = Security(oauth2_scheme)) -> List[str]:
    """Retrieve user roles from OAuth token (placeholder)."""
    # Placeholder: Decode token and fetch roles from identity provider
    return ["user"]  # Default role for testing

async def require_roles(required_roles: List[str], endpoint: str):
    """FastAPI dependency to enforce RBAC."""
    user_roles = await get_current_user_roles()
    if not rbac.check_access(user_roles, endpoint):
        logger.error(f"Access denied to {endpoint} for roles {user_roles}")
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    logger.info(f"Access granted to {endpoint} for roles {user_roles}")
    return user_roles
