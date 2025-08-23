import logging
import toml
from typing import Dict, Any
from pydantic import BaseModel
from httpx import AsyncClient

logger = logging.getLogger(__name__)

class MCPTool(BaseModel):
    name: str
    parameters: Dict[str, Any]

class MCPValidator:
    def __init__(self):
        self.allowlist = toml.load("mcp.toml").get("tools", {}).keys()

    async def validate_tool_call(self, tool: MCPTool) -> bool:
        """Validate MCP tool call against allowlist and parameters."""
        try:
            if tool.name not in self.allowlist:
                raise ValueError(f"Tool {tool.name} not allowed")
            if tool.name == "quantum_sync" and tool.parameters.get("qubits", 0) > 32:
                raise ValueError("Excessive qubit count")
            async with AsyncClient() as client:
                shield_response = await client.post(
                    "https://api.azure.ai/content-safety/prompt-shields",
                    json={"prompt": str(tool.parameters)}
                )
                if shield_response.json().get("malicious"):
                    raise ValueError("Malicious input detected")
            return True
        except Exception as e:
            logger.error(f"Tool validation failed: {str(e)}")
            raise
