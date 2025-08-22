# server/error_handler.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

def setup_error_handlers(app: FastAPI):
    """Set up global error handlers."""
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(
            f"HTTP error for {request.url.path}: "
            f"status={exc.status_code}, detail={exc.detail}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "detail": exc.detail}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unexpected error for {request.url.path}: {str(exc)}"
        )
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "Internal server error"}
        )
