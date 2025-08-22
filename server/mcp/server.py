import asyncio
from mcp.server import Server
from server.mcp.tools import build_tool_list
from server.logging import logger


async def main():
    try:
        server = Server(name="mcp-alchemist")
        for tool in build_tool_list():
            server.add_tool(tool)
        logger.log("Starting MCP server on stdio")
        await server.run_stdio()
    except Exception as e:
        logger.log(f"MCP server error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
