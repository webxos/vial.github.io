from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
async def setup_error_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"message": "Internal server error"})
