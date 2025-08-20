from fastapi import APIRouter, Request, Depends
from mcp.server.fastmcp import FastMCP
from ..sdk.vial_monitor import vial_monitor
from ..security.oauth2 import get_current_user
from ..api.utils import jsonrpc_response

router = APIRouter()
mcp_server = FastMCP()

@router.post("/jsonrpc")
async def monitor_jsonrpc_endpoint(request: Request, user=Depends(get_current_user)):
    data = await request.json()
    method = data.get("method")
    params = data.get("params", {})
    id = data.get("id")

    if method == "monitor/health":
        result = vial_monitor.check_system_health()
        return jsonrpc_response(id, result)
    elif method == "monitor/training":
        result = await vial_monitor.monitor_training()
        return jsonrpc_response(id, result)
    return jsonrpc_response(id, {"error": "Method not found"}, error_code=-32601)
