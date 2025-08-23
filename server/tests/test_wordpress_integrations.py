import pytest
import httpx
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import logging
from server.wordpress.tools.posts import create_post, update_post, get_posts
from server.wordpress.tools.pages import create_page, update_page, get_pages
from server.wordpress.tools.users import create_user, update_user, get_users, delete_user
from server.wordpress.tools.media import upload_media, get_media, delete_media
from server.wordpress.tools.plugins import install_plugin, activate_plugin, deactivate_plugin, uninstall_plugin, get_plugins
from server.wordpress.tools.themes import install_theme, activate_theme, uninstall_theme, get_themes
from server.wordpress.auth import WPAuthConfig

logger = logging.getLogger(__name__)

@pytest.fixture
def wp_config():
    return WPAuthConfig(base_url="http://test.wp.com", username="test", app_password="test", timeout=5)

@pytest.fixture
async def mock_httpx():
    async with AsyncClient() as client:
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = httpx.Response(200, json={"id": 1, "title": "Test"})
            yield mock_request

@pytest.mark.asyncio
async def test_create_post(wp_config, mock_httpx):
    data = {"title": "Test Post", "content": "Content"}
    result = await create_post(data)
    assert result["id"] == 1
    assert mock_httpx.called

@pytest.mark.asyncio
async def test_update_post(wp_config, mock_httpx):
    data = {"id": 1, "title": "Updated"}
    result = await update_post(data)
    assert "updated" in str(result).lower()

@pytest.mark.asyncio
async def test_get_posts(wp_config, mock_httpx):
    query = {"page": 1, "per_page": 5}
    results = await get_posts(query)
    assert len(results) > 0

# Similar tests for pages
@pytest.mark.asyncio
async def test_create_page(wp_config, mock_httpx):
    data = {"title": "Test Page", "content": "Content"}
    result = await create_page(data)
    assert result["id"] == 1

# ... (repeat pattern for update_page, get_pages)

# Users tests
@pytest.mark.asyncio
async def test_create_user(wp_config, mock_httpx):
    data = {"username": "testuser", "email": "test@example.com", "password": "pass"}
    result = await create_user(data)
    assert "id" in result

# ... (update_user, get_users, delete_user)

# Media tests
@pytest.mark.asyncio
async def test_upload_media(wp_config, mock_httpx):
    data = {"file_path": "test.jpg"}
    result = await upload_media(data)
    assert "id" in result

# ... (get_media, delete_media)

# Plugins tests
@pytest.mark.asyncio
async def test_install_plugin(wp_config, mock_httpx):
    data = {"slug": "hello-dolly"}
    result = await install_plugin(data)
    assert "installed" in str(result).lower()

# ... (activate_plugin, deactivate_plugin, uninstall_plugin, get_plugins)

# Themes tests
@pytest.mark.asyncio
async def test_install_theme(wp_config, mock_httpx):
    data = {"slug": "twenty-twenty"}
    result = await install_theme(data)
    assert "installed" in str(result).lower()

# ... (activate_theme, uninstall_theme, get_themes)
