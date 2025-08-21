from redis import Redis
from server.config import get_settings
from server.logging import logger

class RedisHandler:
    def __init__(self):
        self.settings = get_settings()
        self.redis = Redis.from_url(self.settings.REDIS_URL, decode_responses=True)

    def get(self, key: str):
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {str(e)}")
            return None

    def setex(self, key: str, time: int, value: str):
        try:
            self.redis.setex(key, time, value)
        except Exception as e:
            logger.error(f"Redis setex error: {str(e)}")

    def incr(self, key: str):
        try:
            return self.redis.incr(key)
        except Exception as e:
            logger.error(f"Redis incr error: {str(e)}")
            return None

redis_handler = RedisHandler()
