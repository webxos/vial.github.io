from fastapi import Request
from fastapi.responses import JSONResponse
import time


async def cache_response(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    response.headers["X-Cache-Duration"] = f"{duration:.2f}s"
    return response
