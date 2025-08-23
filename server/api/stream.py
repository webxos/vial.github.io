from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid
import json

router = APIRouter(prefix="/v1/stream", tags=["stream"])


@router.get("/task_updates")
async def task_updates(task_id: str, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    
    async def stream_updates():
        try:
            updates = await memory_manager.get_task_updates(task_id, request_id)
            for update in updates:
                yield json.dumps({"update": update, "request_id": request_id}) + "\n"
            logger.info(f"Streamed updates for task {task_id}", request_id=request_id)
        except Exception as e:
            logger.error(f"Stream error: {str(e)}", request_id=request_id)
            raise
    
    return StreamingResponse(stream_updates(), media_type="text/event-stream")
