from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from server.services.advanced_logging import AdvancedLogger
import asyncio


router = APIRouter()
logger = AdvancedLogger()


@router.get("/stream/logs")
async def stream_logs():
    async def log_stream():
        for i in range(10):
            yield f"data: Log event {i} at {asyncio.get_event_loop().time()}\n\n"
            await asyncio.sleep(1)
        logger.log("Log stream completed", extra={"events": 10})
    
    logger.log("Log stream started", extra={"stream": "logs"})
    return StreamingResponse(log_stream(), media_type="text/event-stream")
