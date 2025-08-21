from server.services.mongodb_handler import mongodb_handler
from server.services.redis_handler import redis_handler
from server.logging import logger


async def check_health():
    health_status = {"status": "healthy", "services": {}}
    try:
        mongodb_handler.ping()
        health_status["services"]["mongodb"] = "ok"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        health_status["services"]["mongodb"] = "error"
    try:
        redis_handler.ping()
        health_status["services"]["redis"] = "ok"
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        health_status["services"]["redis"] = "error"
    return health_status
