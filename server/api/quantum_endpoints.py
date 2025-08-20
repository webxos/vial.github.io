from fastapi import APIRouter, Request
from mcp.server.fastmcp import FastMCP
from ..models.webxos_wallet import webxos_wallet
from ..quantum_sync import quantum_sync
from ..api.utils import jsonrpc_response

router = APIRouter()
mcp_server = FastMCP()

@router.post("/jsonrpc")
async def quantum_jsonrpc_endpoint(request: Request):
    data = await request.json()
    method = data.get("method")
    params = data.get("params", {})
    id = data.get("id")

    if method == "wallet/update":
        result = await webxos_wallet.update_wallet(params.get("node_id"), params.get("amount", 0.0))
        return jsonrpc_response(id, result)
    elif method == "wallet/status":
        result = webxos_wallet.get_wallet_status()
        return jsonrpc_response(id, result)
    elif method == "quantum/sync":
        result = await quantum_sync.sync_node(params.get("node_id"), params.get("data", {}))
        return jsonrpc_response(id, result.to_dict())
    return jsonrpc_response(id, {"error": "Method not found"}, error_code=-32601)
