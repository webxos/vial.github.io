import os
import logging
import toml
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

class MCPConfig(BaseModel):
    tools: Dict[str, Dict[str, Any]]
    auth: Dict[str, Dict[str, Any]]

def load_mcp_config() -> MCPConfig:
    """Load and validate MCP configuration from toml and env."""
    try:
        with open("mcp.toml", "r") as f:
            config_data = toml.load(f)
        
        # Merge with env vars
        config_data["auth"]["wordpress"]["base_url"] = os.getenv("WORDPRESS_BASE_URL", config_data["auth"]["wordpress"].get("base_url", ""))
        return MCPConfig(**config_data)
    except Exception as e:
        logger.error(f"Config load failed: {str(e)}")
        raise

def get_tool_handler(tool_name: str) -> str:
    """Get handler for a given tool."""
    config = load_mcp_config()
    tool = config.tools.get(tool_name)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found")
    return tool["handler"]
