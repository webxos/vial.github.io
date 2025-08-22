from fastapi import Request
from server.services.advanced_logging import AdvancedLogger
import time
import hashlib


logger = AdvancedLogger()
cache = {}


async def cache_response(request: Request, call_next):
    cache_key = hashlib.md5(str(request.url).encode()).hexdigest()
    if cache_key in cache and time.time() - cache[cache_key]["timestamp"] < 300:
        logger.log("Cache hit", extra={"url": str(request.url)})
        return cache[cache_key]["response"]
    
    response = await call_next(request)
    cache[cache_key] = {"response": response, "timestamp": time.time()}
    logger.log("Cache miss, stored response", extra={"url": str(request.url)})
    return response
