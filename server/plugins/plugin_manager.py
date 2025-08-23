from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from typing import Dict, Callable
import importlib
import toml
import logging
from fastapi.security import OAuth2AuthorizationCodeBearer

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

class PluginRequest(BaseModel):
    plugin_name: str
    params: Dict

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, Callable] = {}
        self.load_plugins()

    def load_plugins(self):
        """Load plugins from mcp.toml."""
        config = toml.load("mcp.toml")
        for tool, details in config.get("tools", {}).items():
            try:
                module_name, func_name = details["handler"].rsplit(".", 1)
                module = importlib.import_module(module_name)
                self.plugins[tool] = getattr(module, func_name)
                logger.info(f"Loaded plugin: {tool}")
            except Exception as e:
                logger.error(f"Failed to load plugin {tool}: {str(e)}")

    async def execute_plugin(self, plugin_name: str, params: Dict) -> Dict:
        """Execute a plugin with given parameters."""
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return await plugin(params)

plugin_manager = PluginManager()

@router.post("/mcp/plugins")
async def execute_plugin(request: PluginRequest, token: str = Security(oauth2_scheme)):
    """Execute a plugin via API."""
    try:
        result = await plugin_manager.execute_plugin(request.plugin_name, request.params)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Plugin execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
