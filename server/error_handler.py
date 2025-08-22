from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


async def setup_error_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        error_msg = str(exc)
        status_code = 400 if "visual_config" in error_msg.lower() else 500
        content = {"error": "Invalid visual configuration"} if status_code == 400 else {"message": "Internal server error"}
        return JSONResponse(status_code=status_code,
                           content=content)
