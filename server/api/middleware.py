from fastapi import Request, Response
from server.logging import logger

async def logging_middleware(request: Request, call_next):
    logger.log(f"Request: {request.method} {request.url}")
    response: Response = await call_next(request)
    logger.log(f"Response: {response.status_code}")
    return response
