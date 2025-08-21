from fastapi import Request, HTTPException
from server.services.redis_handler import redis_handler
from server.logging import logger


async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    request_count = redis_handler.incr(key)
    if request_count == 1:
        redis_handler.setex(key, 60, request_count)
    if request_count > 100:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(status_code=429, detail="Too many requests")
    response = await call_next(request)
    return response
