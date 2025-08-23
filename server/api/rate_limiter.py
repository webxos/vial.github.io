from fastapi import Depends, HTTPException
from redis import Redis
from server.services.redis_cache import RedisCache
from server.logging_config import logger
import os
import uuid
import time
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")

class RateLimiter:
    def __init__(self):
        self.redis = Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        self.cache = RedisCache()
        self.rate_limit = int(os.getenv("RATE_LIMIT", 100))  # Requests per minute
        self.window = 60  # Time window in seconds

    async def check_rate_limit(self, token: str = Depends(oauth2_scheme)) -> None:
        request_id = str(uuid.uuid4())
        try:
            key = f"rate_limit:{token}"
            current = self.redis.get(key)
            if current is None:
                self.redis.setex(key, self.window, 1)
            elif int(current) >= self.rate_limit:
                logger.warning(f"Rate limit exceeded for token {token}", request_id=request_id)
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            else:
                self.redis.incr(key)
            logger.info(f"Rate limit check passed for token {token}", request_id=request_id)
        except Exception as e:
            logger.error(f"Rate limit error: {str(e)}", request_id=request_id)
            raise HTTPException(status_code=500, detail=str(e))

    async def store_session(self, token: str, session_data: dict, ttl: int = 3600) -> None:
        request_id = str(uuid.uuid4())
        try:
            self.cache.set_cache(f"session:{token}", session_data, ttl)
            logger.info(f"Stored session for token {token}", request_id=request_id)
        except Exception as e:
            logger.error(f"Session store error: {str(e)}", request_id=request_id)
            raise
