from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from server.services.advanced_logging import AdvancedLogger


logger = AdvancedLogger()


def setup_error_handler(app: FastAPI):
    @app.exception_handler(Exception)
    async def custom_exception_handler(request: Request, exc: Exception):
        logger.log("Error occurred",
                   extra={"error": str(exc), "url": str(request.url)})
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )
