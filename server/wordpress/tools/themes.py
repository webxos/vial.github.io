from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from mcp import tool
from .auth import authenticated_request, WPAuthConfig
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
config = WPAuthConfig()

class ThemeInstall(BaseModel):
    slug: str
    version: Optional[str] = "latest"
    activate: bool = True

class ThemeAction(BaseModel):
    stylesheet: str

class ThemeQuery(BaseModel):
    page: int = 1
    per_page: int = 10
    search: Optional[str] = None

@tool("install_wordpress_theme")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def install_theme(data: ThemeInstall) -> Dict[str, Any]:
    """Install and optionally activate a WordPress theme."""
    payload = {"slug": data.slug}
    if data.activate:
        payload["activate"] = True
    return await authenticated_request("POST", "themes", config, json=payload)

@tool("activate_wordpress_theme")
async def activate_theme(data: ThemeAction) -> Dict[str, Any]:
    """Activate a WordPress theme."""
    return await authenticated_request("POST", "themes", config, json={"stylesheet": data.stylesheet, "action": "activate"})

@tool("uninstall_wordpress_theme")
async def uninstall_theme(data: ThemeAction) -> Dict[str, Any]:
    """Uninstall a WordPress theme."""
    return await authenticated_request("DELETE", f"themes/{data.stylesheet}", config)

@tool("get_wordpress_themes")
async def get_themes(query: ThemeQuery) -> List[Dict[str, Any]]:
    """Get installed themes."""
    params = query.dict(exclude_none=True)
    try:
        return await authenticated_request("GET", "themes", config, params=params)
    except Exception as e:
        logger.error(f"Get themes failed: {str(e)}")
        return []
