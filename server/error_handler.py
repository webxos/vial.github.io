from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


async def setup_error_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        if "visual_config" in str(exc):
            return JSONResponse(status_code=400,
                               content={"error": "Invalid visual configuration"})
        return JSONResponse(status_code=500,
                           content={"message": "Internal server error"})
