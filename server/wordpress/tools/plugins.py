from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from mcp import tool
from .auth import authenticated_request, WPAuthConfig
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
config = WPAuthConfig()

class PluginInstall(BaseModel):
    slug: str
    version: Optional[str] = "latest"
    activate: bool = True

class PluginAction(BaseModel):
    slug: str

class PluginQuery(BaseModel):
    page: int = 1
    per_page: int = 10
    search: Optional[str] = None
    status: Optional[str] = None  # active/inactive

@tool("install_wordpress_plugin")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def install_plugin(data: PluginInstall) -> Dict[str, Any]:
    """Install and optionally activate a WordPress plugin."""
    payload = {"slug": data.slug, "status": "install"}
    if data.activate:
        payload["activate"] = True
    return await authenticated_request("POST", "plugins", config, json=payload)

@tool("activate_wordpress_plugin")
async def activate_plugin(data: PluginAction) -> Dict[str, Any]:
    """Activate a WordPress plugin."""
    return await authenticated_request("POST", f"plugins/{data.slug}", config, json={"status": "active"})

@tool("deactivate_wordpress_plugin")
async def deactivate_plugin(data: PluginAction) -> Dict[str, Any]:
    """Deactivate a WordPress plugin."""
    return await authenticated_request("POST", f"plugins/{data.slug}", config, json={"status": "inactive"})

@tool("uninstall_wordpress_plugin")
async def uninstall_plugin(data: PluginAction) -> Dict[str, Any]:
    """Uninstall a WordPress plugin."""
    return await authenticated_request("DELETE", f"plugins/{data.slug}", config)

@tool("get_wordpress_plugins")
async def get_plugins(query: PluginQuery) -> List[Dict[str, Any]]:
    """Get installed plugins with filters."""
    params = query.dict(exclude_none=True)
    try:
        return await authenticated_request("GET", "plugins", config, params=params)
    except Exception as e:
        logger.error(f"Get plugins failed: {str(e)}")
        return []
