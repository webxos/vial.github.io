from fastapi import APIRouter, Request, Depends
from mcp.server.fastmcp import FastMCP
from ..models.mcp_alchemist import mcp_alchemist
from ..security.oauth2 import get_current_user
from ..api.utils import jsonrpc_response

router = APIRouter()
mcp_server = FastMCP()

@router.post("/jsonrpc")
async def alchemist_jsonrpc_endpoint(request: Request, user=Depends(get_current_user)):
    data = await request.json()
    method = data.get("method")
    params = data.get("params", {})
    id = data.get("id")

    if method == "alchemist/train":
        result = await mcp_alchemist.train_wallet(params.get("data", ""), user)
        return jsonrpc_response(id, result)
    elif method == "alchemist/translate":
        result = mcp_alchemist.translate_script(params.get("lang", ""), params.get("script", ""))
        return jsonrpc_response(id, result)
    elif method == "alchemist/diagnose":
        result = mcp_alchemist.diagnose_script(params.get("script", ""))
        return jsonrpc_response(id, result)
    elif method == "alchemist/git_train":
        result = await mcp_alchemist.git_train(params.get("commit_msg", ""), user)
        return jsonrpc_response(id, result)
    elif method == "alchemist/create_wallet":
        result = mcp_alchemist.create_new_wallet(user)
        return jsonrpc_response(id, result)
    return jsonrpc_response(id, {"error": "Method not found", "sqlite_error": "Unknown method"}, error_code=-32601)
