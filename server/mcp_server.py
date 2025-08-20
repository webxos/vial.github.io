from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, Resource, Prompt
from .api.endpoints import router
from .config import config
from .logging import logger

class VialMCPServer:
    def __init__(self):
        self.mcp = FastMCP(
            transport="stdio",
            port=config.API_PORT,
            debug=config.DEBUG
        )
        self.mcp.register_router(router)
        self.setup_primitives()

    def setup_primitives(self):
        # Initial Tool primitive (expandable)
        self.mcp.register_tool(Tool(
            name="quantum_link",
            description="Establish quantum link between nodes",
            input_schema={"type": "object", "properties": {"node_a": {"type": "string"}, "node_b": {"type": "string"}}},
            handler=lambda params: {"status": "established", "link_id": f"{params['node_a']}-{params['node_b']}"}
        ))
        logger.info("MCP server initialized with Tool primitive")

    def run(self):
        self.mcp.run()

if __name__ == "__main__":
    server = VialMCPServer()
    server.run()
