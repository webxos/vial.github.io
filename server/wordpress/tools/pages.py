from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from mcp import tool
from .auth import authenticated_request, WPAuthConfig

logger = logging.getLogger(__name__)
config = WPAuthConfig()

class PageCreate(BaseModel):
    title: str
    content: str
    status: str = "draft"
    parent: Optional[int] = None
    menu_order: Optional[int] = 0
    meta: Optional[Dict[str, Any]] = None

class PageUpdate(BaseModel):
    id: int
    title: Optional[str] = None
    content: Optional[str] = None
    status: Optional[str] = None
    parent: Optional[int] = None
    menu_order: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

class PageQuery(BaseModel):
    page: int = 1
    per_page: int = 10
    search: Optional[str] = None
    include_meta: bool = True

@tool("create_wordpress_page")
async def create_page(data: PageCreate) -> Dict[str, Any]:
    """Create a WordPress page."""
    payload = data.dict(exclude_none=True)
    return await authenticated_request("POST", "pages", config, json=payload)

@tool("update_wordpress_page")
async def update_page(data: PageUpdate) -> Dict[str, Any]:
    """Update a WordPress page."""
    payload = data.dict(exclude={"id"}, exclude_none=True)
    return await authenticated_request("POST", f"pages/{data.id}", config, json=payload)

@tool("get_wordpress_pages")
async def get_pages(query: PageQuery) -> List[Dict[str, Any]]:
    """Get WordPress pages with filters."""
    params = query.dict(exclude_none=True)
    if query.include_meta:
        params["_embed"] = "wp:featuredmedia"
    try:
        return await authenticated_request("GET", "pages", config, params=params)
    except Exception as e:
        logger.error(f"Get pages failed: {str(e)}")
        return []
