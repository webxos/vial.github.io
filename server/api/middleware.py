# server/api/middleware.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from server.services.advanced_logging import AdvancedLogger
import os

class VercelMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """Validate requests with Vercel token and log events."""
        try:
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            if token != os.getenv("VERCEL_TOKEN"):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid Vercel token"
                )
            logger = AdvancedLogger()
            await logger.log_event(
                event=f"Request to {request.url.path}",
                wallet_address="e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d"
            )
            response = await call_next(request)
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
