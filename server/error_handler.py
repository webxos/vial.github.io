from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from server.logging import logger


async def exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {str(exc)}")
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
