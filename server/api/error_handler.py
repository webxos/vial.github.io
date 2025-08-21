from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from server.services.logging import Logger


def setup_error_handlers(app: FastAPI):
    logger = Logger("error_handler")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        await logger.error(f"HTTP error: {exc.detail}", user_id="system")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.status_code, "message": exc.detail}}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        await logger.error(f"Unexpected error: {str(exc)}", user_id="system")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": 500, "message": "Internal server error"}}
        )
