from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from mcp import tool  # MCP SDK decorator
from .auth import authenticated_request, WPAuthConfig

logger = logging.getLogger(__name__)
config = WPAuthConfig()

class PostCreate(BaseModel):
    title: str
    content: str
    status: str = "draft"
    categories: Optional[List[int]] = None
    featured_media: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

class PostUpdate(BaseModel):
    id: int
    title: Optional[str] = None
    content: Optional[str] = None
    status: Optional[str] = None
    categories: Optional[List[int]] = None
    featured_media: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

class PostQuery(BaseModel):
    page: int = 1
    per_page: int = 10
    search: Optional[str] = None
    categories: Optional[List[int]] = None
    include_meta: bool = True

@tool("create_wordpress_post")
async def create_post(data: PostCreate) -> Dict[str, Any]:
    """Create a WordPress post via REST API."""
    payload = data.dict(exclude_none=True)
    return await authenticated_request("POST", "posts", config, json=payload)

@tool("update_wordpress_post")
async def update_post(data: PostUpdate) -> Dict[str, Any]:
    """Update a WordPress post with partial fields."""
    payload = data.dict(exclude={"id"}, exclude_none=True)
    return await authenticated_request("POST", f"posts/{data.id}", config, json=payload)

@tool("get_wordpress_posts")
async def get_posts(query: PostQuery) -> List[Dict[str, Any]]:
    """Get WordPress posts with filtering and pagination."""
    params = query.dict(exclude_none=True)
    if query.include_meta:
        params["_embed"] = "wp:featuredmedia,wp:term"
    try:
        return await authenticated_request("GET", "posts", config, params=params)
    except Exception as e:
        logger.error(f"Get posts failed: {str(e)}")
        return []
