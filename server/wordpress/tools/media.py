from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from mcp import tool
from .auth import authenticated_request, WPAuthConfig
import httpx  # For multipart

logger = logging.getLogger(__name__)
config = WPAuthConfig()

class MediaUpload(BaseModel):
    file_path: str
    title: Optional[str] = None
    alt_text: Optional[str] = None

class MediaQuery(BaseModel):
    page: int = 1
    per_page: int = 10
    search: Optional[str] = None

@tool("upload_wordpress_media")
async def upload_media(data: MediaUpload) -> Dict[str, Any]:
    """Upload media to WordPress library."""
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        with open(data.file_path, "rb") as f:
            files = {"file": (data.file_path.split("/")[-1], f)}
            headers = {"Content-Disposition": f'attachment; filename="{files["file"][0]}"'}
            response = await client.post(f"{config.base_url}/wp-json/wp/v2/media", files=files, headers=headers, auth=(config.username, config.app_password))
            response.raise_for_status()
            return response.json()

@tool("get_wordpress_media")
async def get_media(query: MediaQuery) -> List[Dict[str, Any]]:
    """Get media items."""
    params = query.dict(exclude_none=True)
    try:
        return await authenticated_request("GET", "media", config, params=params)
    except Exception as e:
        logger.error(f"Get media failed: {str(e)}")
        return []

@tool("delete_wordpress_media")
async def delete_media(media_id: int) -> Dict[str, Any]:
    """Delete media item."""
    return await authenticated_request("DELETE", f"media/{media_id}?force=true", config)
