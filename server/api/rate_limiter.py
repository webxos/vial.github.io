from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from server.services.redis_handler import redis_handler
from server.logging import logger
import time

async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    limit = 100  # Requests per minute
    window = 60  # Seconds

    try:
        # Check rate limit in Redis
        current = redis_handler.get(key)
        if current is None:
            redis_handler.setex(key, window, 1)
        elif int(current) >= limit:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"jsonrpc": "2.0", "error": {"code": 429, "message": "Rate limit exceeded"}, "id": None}
            )
        else:
            redis_handler.incr(key)
        
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Rate limiter error: {str(e)}")
        raise HTTPException(status_code=500, detail="Rate limiter error")
