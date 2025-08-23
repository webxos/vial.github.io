from fastapi import HTTPException, Depends
from server.services.redis_cache import RedisCache
from server.logging_config import logger
import uuid

class RateLimiter:
    def __init__(self):
        self.redis = RedisCache()

    async def check_rate_limit(self, token: str) -> None:
        request_id = str(uuid.uuid4())
        try:
            key = f"rate_limit:{token}"
            count = await self.redis.get(key)
            count = int(count) if count else 0
            if count >= 100:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            await self.redis.set(key, count + 1, ex=3600)
            logger.info(f"Rate limit check passed for token {token}", request_id=request_id)
        except Exception as e:
            logger.error(f"Rate limit check error: {str(e)}", request_id=request_id)
            raise
