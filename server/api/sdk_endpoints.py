from fastapi import APIRouter, Request, Depends
from mcp.server.fastmcp import FastMCP
from ..sdk.vial_sdk import vial_sdk
from ..security.oauth2 import get_current_user
from ..api.utils import jsonrpc_response

router = APIRouter()
mcp_server = FastMCP()

@router.post("/jsonrpc")
async def sdk_jsonrpc_endpoint(request: Request, user=Depends(get_current_user)):
    data = await request.json()
    method = data.get("method")
    params = data.get("params", {})
    id = data.get("id")

    if method == "sdk/initialize":
        result = await vial_sdk.initialize_system(user)
        return jsonrpc_response(id, result)
    elif method == "sdk/train_and_sync":
        result = await vial_sdk.train_and_sync(params.get("data", ""), user)
        return jsonrpc_response(id, result)
    elif method == "sdk/deploy":
        result = vial_sdk.deploy_container(params.get("image_tag", "vial_mcp_sdk:latest"))
        return jsonrpc_response(id, result)
    return jsonrpc_response(id, {"error": "Method not found"}, error_code=-32601)
