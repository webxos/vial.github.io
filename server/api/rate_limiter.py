# server/api/rate_limiter.py
from fastapi import Request, HTTPException
import redis
from server.config.settings import settings
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
redis_client = redis.Redis(host=settings.REDIS_HOST, port=6379, db=0)

async def rate_limit(request: Request, call_next):
    """Rate limit requests based on client IP or wallet address."""
    client_ip = request.client.host
    wallet_address = request.headers.get("X-Wallet-Address", client_ip)
    key = f"rate_limit:{wallet_address}"
    
    current_count = redis_client.get(key)
    if current_count and int(current_count) >= 100:
        logger.warning(f"Rate limit exceeded for {wallet_address}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await call_next(request)
    redis_client.incr(key)
    redis_client.expireat(key, int((datetime.now() + timedelta(minutes=1)).timestamp()))
    return response
