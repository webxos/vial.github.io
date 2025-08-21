from fastapi import Request
from server.services.redis_handler import redis_handler
from server.logging import logger
import hashlib
import json

async def cache_middleware(request: Request, call_next):
    # Generate cache key based on request URL and method
    cache_key = f"cache:{hashlib.md5(f'{request.method}:{request.url}'.encode()).hexdigest()}"
    
    try:
        # Check if response is cached
        cached_response = redis_handler.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit for {cache_key}")
            return json.loads(cached_response)
        
        # Proceed with request and cache response
        response = await call_next(request)
        if response.status_code == 200:
            response_body = json.dumps(response.body.decode())
            redis_handler.setex(cache_key, 300, response_body)  # Cache for 5 minutes
            logger.info(f"Cache set for {cache_key}")
        return response
    except Exception as e:
        logger.error(f"Cache error: {str(e)}")
        return await call_next(request)
