from fastapi import Request
from server.services.advanced_logging import AdvancedLogger
import time


logger = AdvancedLogger()


async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.log("Request processed", extra={
        "method": request.method,
        "url": str(request.url),
        "duration_ms": round(duration * 1000, 2)
    })
    return response
