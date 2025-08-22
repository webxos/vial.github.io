from fastapi import Response
from fastapi.responses import JSONResponse
from functools import wraps
import redis
import json
from server.config import settings
from server.logging import logger

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    decode_responses=True
)


def cache_response(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        cache_key = f"cache:{func.__name__}:{json.dumps(kwargs)}"
        cached = redis_client.get(cache_key)
        if cached:
            logger.log(f"Cache hit for {cache_key}")
            return JSONResponse(content=json.loads(cached))
        response = await func(*args, **kwargs)
        if isinstance(response, Response):
            content = response.body.decode()
            redis_client.setex(cache_key, settings.CACHE_TTL, content)
            logger.log(f"Cache set for {cache_key}")
        return response
    return wrapper
