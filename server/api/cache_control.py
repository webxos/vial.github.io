# server/api/cache_control.py
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import redis
from server.config.settings import settings
import logging

logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=settings.REDIS_HOST, port=6379, db=0)

async def cache_response(request: Request, call_next):
    """Middleware to cache API responses."""
    cache_key = f"cache:{request.url.path}:{request.query_params}"
    cached_response = redis_client.get(cache_key)
    
    if cached_response:
        logger.info(f"Cache hit for {request.url.path}")
        return JSONResponse(content=cached_response.decode('utf-8'))
    
    response = await call_next(request)
    if response.status_code == 200:
        redis_client.setex(cache_key, 300, response.body)
        logger.info(f"Cached response for {request.url.path}")
    
    return response
