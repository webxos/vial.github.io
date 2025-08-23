from typing import List, Dict, Any, Optional
from pydantic import BaseModel, EmailStr
import logging
from mcp import tool
from .auth import authenticated_request, WPAuthConfig

logger = logging.getLogger(__name__)
config = WPAuthConfig()

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    roles: List[str] = ["subscriber"]

class UserUpdate(BaseModel):
    id: int
    email: Optional[EmailStr] = None
    roles: Optional[List[str]] = None
    password: Optional[str] = None

class UserQuery(BaseModel):
    page: int = 1
    per_page: int = 10
    search: Optional[str] = None

@tool("create_wordpress_user")
async def create_user(data: UserCreate) -> Dict[str, Any]:
    """Create a WordPress user."""
    payload = data.dict()
    return await authenticated_request("POST", "users", config, json=payload)

@tool("update_wordpress_user")
async def update_user(data: UserUpdate) -> Dict[str, Any]:
    """Update a WordPress user."""
    payload = data.dict(exclude={"id"}, exclude_none=True)
    return await authenticated_request("POST", f"users/{data.id}", config, json=payload)

@tool("get_wordpress_users")
async def get_users(query: UserQuery) -> List[Dict[str, Any]]:
    """Get WordPress users."""
    params = query.dict(exclude_none=True)
    try:
        return await authenticated_request("GET", "users", config, params=params)
    except Exception as e:
        logger.error(f"Get users failed: {str(e)}")
        return []

@tool("delete_wordpress_user")
async def delete_user(user_id: int) -> Dict[str, Any]:
    """Delete a WordPress user."""
    return await authenticated_request("DELETE", f"users/{user_id}?force=true", config)
