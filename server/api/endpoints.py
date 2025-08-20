from fastapi import APIRouter, Request
from mcp.server.fastmcp import FastMCP
from ..security import get_current_user
from ..error_handler import handle_sqlite_error
from ..services.database import get_db
from ..models.alchemy_pytorch import AlchemyPyTorch
from ..api.utils import jsonrpc_response

router = APIRouter()
mcp_server = FastMCP()
model = AlchemyPyTorch()

@router.post("/jsonrpc")
@handle_sqlite_error
async def jsonrpc_endpoint(request: Request, db=None, user=Depends(get_current_user)):
    data = await request.json()
    method = data.get("method")
    params = data.get("params", {})
    id = data.get("id")

    if method == "tools/list":
        tools = [{"name": "quantum_link", "description": "Establish quantum link", "inputSchema": {"type": "object", "properties": {"node_a": {"type": "string"}, "node_b": {"type": "string"}}}}]
        return jsonrpc_response(id, tools)
    elif method == "tools/call":
        if params.get("name") == "quantum_link":
            result = model.establish_quantum_link(params.get("node_a"), params.get("node_b"))
            return jsonrpc_response(id, result)
    return jsonrpc_response(id, {"error": "Method not found"}, error_code=-32601)
