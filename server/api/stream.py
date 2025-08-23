from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from server.mcp.auth import oauth2_scheme
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid
import json

router = APIRouter()

@router.get("/stream/tasks")
async def stream_tasks(token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        async def task_stream():
            alchemist = Alchemist()
            tasks = ["vial_status_get", "quantum_circuit_build", "git_commit_push"]
            for task in tasks:
                result = await alchemist.delegate_task(task, {"params": {}})
                yield f"data: {json.dumps({'task': task, 'result': result, 'request_id': request_id})}\n\n"
                logger.log(f"Streamed task: {task}", request_id=request_id)
        return StreamingResponse(task_stream(), media_type="text/event-stream")
    except Exception as e:
        logger.log(f"Stream error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
