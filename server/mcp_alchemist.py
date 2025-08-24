```python
import asyncio
import os
import httpx
from mcp.types import Tool, MCPRequest
from mcp.server import Server
from .services.spacex_service import SpaceXService
from .database.session import session_scope
from .models.base import QuantumCircuit, NASADataset

class MCPAlchemist:
    def __init__(self, server: Server):
        self.server = server
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.buildable_api_key = os.getenv("BUILDABLE_API_KEY")
        self.buildable_project_id = os.getenv("BUILDABLE_PROJECT_ID")
        self.register_tools()

    def register_tools(self):
        @self.server.tool(name="github_create_issue", description="Create GitHub issue")
        async def github_create_issue(title: str, body: str) -> dict:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.github.com/repos/webxos/vial.github.io/issues",
                    json={"title": title, "body": body},
                    headers={"Authorization": f"Bearer {self.github_token}"}
                )
                return response.json()

        @self.server.tool(name="buildable_get_task", description="Get next Buildable task")
        async def buildable_get_task() -> dict:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://bldbl.dev/api/projects/{self.buildable_project_id}/tasks/next",
                    headers={"Authorization": f"Bearer {self.buildable_api_key}"}
                )
                return response.json()

        @self.server.tool(name="store_quantum_circuit", description="Store quantum circuit with wallet")
        async def store_quantum_circuit(qasm: str, wallet_id: str) -> dict:
            with session_scope() as session:
                circuit = QuantumCircuit(qasm_code=qasm, wallet_id=wallet_id)
                session.add(circuit)
                return {"status": "Circuit stored", "id": circuit.id}

    async def process_request(self, request: MCPRequest):
        return await self.server.process_request(request)
```
