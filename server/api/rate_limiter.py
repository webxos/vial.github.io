from fastapi import Request, HTTPException
from server.services.redis_handler import redis_client
import time


async def rate_limit(request: Request, call_next):
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    limit = 100  # requests per minute
    window = 60  # seconds

    count = await redis_client.get(key)
    count = int(count) if count else 0

    if count >= limit:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )

    response = await call_next(request)
    await redis_client.incr(key)
    await redis_client.expire(key, window)
    return response
