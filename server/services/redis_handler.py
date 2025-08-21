from redis import Redis
from server.config import get_settings
from server.logging import logger


class RedisHandler:
    def __init__(self):
        settings = get_settings()
        self.redis = Redis.from_url(settings.REDIS_URL)


    def get(self, key: str):
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Redis get failed: {str(e)}")
            return None


    def setex(self, key: str, time: int, value: str):
        try:
            self.redis.setex(key, time, value)
        except Exception as e:
            logger.error(f"Redis setex failed: {str(e)}")


    def incr(self, key: str):
        try:
            return self.redis.incr(key)
        except Exception as e:
            logger.error(f"Redis incr failed: {str(e)}")
            return 0


    def ping(self):
        try:
            self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {str(e)}")
            return False


redis_handler = RedisHandler()
