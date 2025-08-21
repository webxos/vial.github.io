from fastapi import Request
from server.services.redis_handler import redis_handler
from server.logging import logger
import hashlib
import json


async def cache_middleware(request: Request, call_next):
    cache_key = (f"cache:{hashlib.md5(f'{request.method}:{request.url}'"
                 f".encode()).hexdigest()}")
    try:
        cached_response = redis_handler.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit for {cache_key}")
            return json.loads(cached_response)
        response = await call_next(request)
        if response.status_code == 200:
            response_body = json.dumps(response.body.decode())
            redis_handler.setex(cache_key, 300, response_body)
            logger.info(f"Cache set for {cache_key}")
        return response
    except Exception as e:
        logger.error(f"Cache error: {str(e)}")
        return await call_next(request)
