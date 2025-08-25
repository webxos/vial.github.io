from mcp.server import Server
from mcp.types import Tool, Resource
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import asyncio
from typing import List, Dict
from server.services.auth_service import verify_token
from server.services.quantum_service import QuantumService

app = FastAPI()
mcp_server = Server("webxos-vial-mcp")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
quantum_service = QuantumService()

@mcp_server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(name="quantum_sync", description="Synchronize quantum circuit data"),
        Tool(name="svg_render", description="Render SVG for video processing"),
        Tool(name="wallet_balance", description="Check .md wallet balance")
    ]

@mcp_server.resource("quantum_network")
async def quantum_network() -> Resource:
    return Resource(name="quantum_network", data={"status": "active", "qubits": []})

@app.get("/mcp/health")
async def health_check():
    return {"status": "healthy", "mcp_version": "2025-03-26"}

@app.post("/mcp/tools/quantum_sync")
async def quantum_sync(params: Dict, token: str = Depends(oauth2_scheme)):
    verify_token(token)
    try:
        result = await quantum_service.sync_circuit(params.get("circuit_id"))
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/tools/wallet_balance")
async def get_wallet_balance(wallet_id: str, token: str = Depends(oauth2_scheme)):
    verify_token(token)
    balance = await quantum_service.get_wallet_balance(wallet_id)
    return {"wallet_id": wallet_id, "balance": balance}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
