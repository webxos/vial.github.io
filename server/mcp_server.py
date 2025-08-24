```python
import uvicorn
from fastapi import FastAPI
from .config import Config
from .api.fastapi_router import app as fastapi_app
from .agents.astronomy import AstronomyAgent
from prometheus_client import Counter, Gauge
from modelcontextprotocol import MCPClient, TransportWebSocket

server_starts_total = Counter('mcp_server_starts_total', 'Total server starts')
server_active = Gauge('mcp_server_active', 'Server active status')
astronomy_tasks_total = Counter('mcp_astronomy_tasks_total', 'Total astronomy agent tasks')
gibs_requests_total = Counter('mcp_gibs_requests_total', 'Total GIBS requests')

class VialMCPServer:
    def __init__(self):
        self.config = Config()
        self.app = fastapi_app
        self.server = MCPClient(transport=TransportWebSocket(host=self.config.host, port=self.config.port))
        self.astronomy_agent = AstronomyAgent()

    async def list_tools(self):
        return await self.server.list_tools()

    async def process_request(self, request: dict):
        if request.get('tool') == 'astronomy_data':
            astronomy_tasks_total.inc()
            return await self.astronomy_agent.fetch_data(request.get('args', {}))
        if request.get('tool') == 'gibs_data':
            gibs_requests_total.inc()
            return await self.astronomy_agent.fetch_gibs_data(request.get('args', {}))
        return await self.server.process_request(request)

    async def start(self):
        server_starts_total.inc()
        server_active.set(1)
        try:
            await uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info"
            )
        finally:
            server_active.set(0)

if __name__ == "__main__":
    import asyncio
    server = VialMCPServer()
    asyncio.run(server.start())
```
