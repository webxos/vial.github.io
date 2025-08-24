```python
from mcp.server import Server
from mcp.types import Tool, Resource
import qiskit
import asyncio

class VialMCPServer:
    def __init__(self):
        self.server = Server("vial-mcp")
        self.register_tools()

    def register_tools(self):
        @self.server.tool(name="quantum_sync", description="Run Qiskit quantum circuit")
        async def quantum_sync(qasm: str) -> dict:
            try:
                circuit = qiskit.QuantumCircuit.from_qasm_str(qasm)
                backend = qiskit.Aer.get_backend('qasm_simulator')
                result = qiskit.execute(circuit, backend, shots=1024).result()
                return {"counts": result.get_counts()}
            except Exception as e:
                return {"error": str(e)}

        @self.server.resource(name="nasa_data", description="Query NASA datasets")
        async def nasa_data(query: str) -> list:
            # Placeholder for NASA API integration
            return [{"dataset_id": "example", "title": f"Mock NASA Data: {query}"}]

    async def start(self):
        await self.server.start(transport="websocket", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    server = VialMCPServer()
    asyncio.run(server.start())
