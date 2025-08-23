from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
from server.mcp.auth import oauth2_scheme
import uuid


router = APIRouter()


class InspectorRequest(BaseModel):
    request_id: str


@router.post("/mcp/inspector")
async def inspect_task(
    request: InspectorRequest,
    token: str = Depends(oauth2_scheme),
    alchemist: Alchemist = Depends(Alchemist)
):
    request_id = str(uuid.uuid4())
    try:
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        prompt_history = await alchemist.get_prompt_history(request.request_id)
        errors = list(alchemist.db.errors.find({"request_id": request.request_id}))
        logger.log(f"Inspected task: {request.request_id}", request_id=request_id)
        return {
            "status": "success",
            "prompt_history": prompt_history,
            "errors": errors,
            "request_id": request_id
        }
    except Exception as e:
        logger.log(f"Inspector error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
