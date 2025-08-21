from fastapi import APIRouter, Request
from server.utils import parse_jsonrpc_request, build_jsonrpc_response
from server.services.git_trainer import GitTrainer
from server.models.mcp_alchemist import MCPAlchemist


router = APIRouter()


@router.post("/")
async def jsonrpc_endpoint(request: Request):
    try:
        data = await request.json()
        method, params, id = parse_jsonrpc_request(data)
        if method == "create_repo":
            git_trainer = GitTrainer()
            result = await git_trainer.execute_task("create_repo", params)
            return build_jsonrpc_response(id, result=result)
        elif method == "predict_quantum":
            alchemist = MCPAlchemist()
            result = await alchemist.predict_quantum_outcome(params)
            return build_jsonrpc_response(id, result=result)
        else:
            return build_jsonrpc_response(
                id,
                error={"code": -32601, "message": "Method not found"}
            )
    except Exception as e:
        return build_jsonrpc_response(
            id=data.get("id"),
            error={"code": -32000, "message": str(e)}
        )
