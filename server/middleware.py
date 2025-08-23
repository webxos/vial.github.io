import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi_limiter import FastAPILimiter
from redis.asyncio import Redis
from httpx import AsyncClient

logger = logging.getLogger(__name__)

async def prompt_shield_middleware(request: Request, call_next):
    """Middleware for Prompt Shields and rate limiting."""
    try:
        body = await request.json()
        async with AsyncClient() as client:
            shield_response = await client.post(
                "https://api.azure.ai/content-safety/prompt-shields",
                json={"prompt": str(body)}
            )
            if shield_response.json().get("malicious"):
                raise HTTPException(status_code=400, detail="Malicious input detected")
        
        redis = Redis.from_url("redis://localhost:6379/0")
        identifier = request.client.host
        if not await FastAPILimiter.check_rate_limit(identifier, redis, limit=10, window=60):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
    except Exception as e:
        logger.error(f"Middleware error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
