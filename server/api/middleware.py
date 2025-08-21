from fastapi import Request
from fastapi.responses import JSONResponse
from server.logging import logger

async def cors_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    logger.info(f"CORS headers added for {request.url}")
    return response
