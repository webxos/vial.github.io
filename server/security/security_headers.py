from fastapi import FastAPI
from fastapi.responses import Response
from server.services.advanced_logging import AdvancedLogger


def setup_security_headers(app: FastAPI):
    logger = AdvancedLogger()

    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        logger.log("Security headers added", extra={"path": request.url.path})
        return response
