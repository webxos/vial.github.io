import asyncio
import json
import sys
import uuid
from fastapi import FastAPI
from server.services.mcp_alchemist import setup_mcp_alchemist
from server.mcp.tools import build_tool_list
from server.mcp.auth import map_oauth_to_mcp_session
from server.logging import logger


app = FastAPI(title="MCP Alchemist Server")


class Server:
    def __init__(self, name: str):
        self.name = name
        self.tools = []
        self.auth_handler = None

    def add_tool(self, tool, auth_handler=None):
        self.tools.append(tool)
        self.auth_handler = auth_handler

    async def run_stdio(self):
        request_id = str(uuid.uuid4())
        logger.log(f"Starting MCP server {self.name} on stdio", request_id=request_id)
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                data = json.loads(line.strip())
                await self.handle_request(data, request_id)
            except Exception as e:
                logger.log(f"Stdio error: {str(e)}", request_id=request_id)
                print(json.dumps({"error": str(e), "request_id": request_id}))

    async def handle_request(self, data: dict, request_id: str):
        tool_name = data.get("tool")
        params = data.get("params", {})
        token = data.get("token", "")
        if tool_name not in [tool.name for tool in self.tools]:
            logger.log(f"Tool {tool_name} not found", request_id=request_id)
            print(json.dumps({"error": f"Tool {tool_name} not found", "request_id": request_id}))
            return
        try:
            if self.auth_handler:
                await self.auth_handler(token, request_id)
            tool = self.tools[[t.name for t in self.tools].index(tool_name)]
            result = await tool.handler(params)
            print(json.dumps({"status": "success", "result": result, "request_id": request_id}))
        except Exception as e:
            logger.log(f"Tool {tool_name} error: {str(e)}", request_id=request_id)
            print(json.dumps({"error": str(e), "request_id": request_id}))


async def main():
    request_id = str(uuid.uuid4())
    server = Server(name="mcp-alchemist")
    setup_mcp_alchemist(app)
    for tool in build_tool_list():
        server.add_tool(tool, auth_handler=lambda token, req_id: map_oauth_to_mcp_session(token, req_id))
    if sys.stdin.isatty():
        logger.log("Running MCP server on WebSocket", request_id=request_id)
        import uvicorn
        await uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
