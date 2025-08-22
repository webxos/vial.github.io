from fastapi import FastAPI, Request
from server.services.advanced_logging import AdvancedLogger
import time


logger = AdvancedLogger()
rate_limits = {}


def setup_rate_limiter(app: FastAPI):
    @app.middleware("http")
    async def limit_requests(request: Request, call_next):
        client_ip = request.client.host
        if client_ip not in rate_limits:
            rate_limits[client_ip] = {"count": 0, "timestamp": time.time()}
        
        if time.time() - rate_limits[client_ip]["timestamp"] < 60:
            rate_limits[client_ip]["count"] += 1
            if rate_limits[client_ip]["count"] > 100:
                logger.log("Rate limit exceeded",
                           extra={"client_ip": client_ip})
                return {"error": "Rate limit exceeded"}
        else:
            rate_limits[client_ip] = {"count": 1, "timestamp": time.time()}
        
        response = await call_next(request)
        logger.log("Request processed",
                   extra={"client_ip": client_ip})
        return response
