import redis.asyncio as redis
from server.config import settings


redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


async def test_redis_connection():
    try:
        await redis_client.ping()
        return {"status": "connected"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
