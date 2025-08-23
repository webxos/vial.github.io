from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from server.api import (
    auth, endpoints, quantum_endpoints, websocket,
    jsonrpc, void, troubleshoot, help, comms_hub, upload, stream, visual_router, webxos_wallet
)
from server.services.mcp_alchemist import setup_mcp_alchemist
from server.services.backup_restore import BackupRestoreService
from server.services.vial_manager import VialManager
from server.mcp.auth import map_oauth_to_mcp_session
from server.mcp.tools import build_tool_list
from server.logging import logger
import asyncio
import json
import sys
import uuid
import os
from fastapi.security import OAuth2PasswordBearer


app = FastAPI(
    title="Vial MCP Controller",
    description="Modular control plane for AI-driven task management",
    version="2.9.3",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


async def setup_services():
    setup_mcp_alchemist(app)
    app.include_router(auth.router, prefix="/auth")
    app.include_router(endpoints.router)
    app.include_router(quantum_endpoints.router, prefix="/quantum")
    app.include_router(websocket.router)
    app.include_router(jsonrpc.router)
    app.include_router(void.router)
    app.include_router(troubleshoot.router)
    app.include_router(help.router)
    app.include_router(comms_hub.router)
    app.include_router(upload.router)
    app.include_router(stream.router)
    app.include_router(visual_router.router, prefix="/visual")
    app.include_router(webxos_wallet.router, prefix="/wallet")


async def main():
    request_id = str(uuid.uuid4())
    await setup_services()
    server = Server(name="mcp-alchemist")
    for tool in build_tool_list():
        server.add_tool(tool, auth_handler=lambda token, req_id: map_oauth_to_mcp_session(token, req_id))
    if os.isatty(sys.stdin.fileno()):
        logger.log("Running MCP server on WebSocket", request_id=request_id)
        import uvicorn
        await uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
