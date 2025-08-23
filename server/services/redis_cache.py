from redis import Redis
import json
from server.logging_config import logger
import os
import uuid

class RedisCache:
    def __init__(self):
        self.redis = Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )

    def get_cache(self, key: str) -> dict:
        request_id = str(uuid.uuid4())
        try:
            cached = self.redis.get(key)
            if cached:
                logger.info(f"Cache hit for {key}", request_id=request_id)
                return json.loads(cached)
            logger.info(f"Cache miss for {key}", request_id=request_id)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}", request_id=request_id)
            return None

    def set_cache(self, key: str, value: dict, ttl: int = 3600) -> None:
        request_id = str(uuid.uuid4())
        try:
            self.redis.setex(key, ttl, json.dumps(value))
            logger.info(f"Cache set for {key}", request_id=request_id)
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}", request_id=request_id)
