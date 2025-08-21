from server.services.mongodb_handler import mongodb_handler
from server.services.redis_handler import redis_handler
from server.logging import logger


def check_system_health():
    health_status = {"status": "healthy", "services": {}}
    
    try:
        mongodb_handler.db.command("ping")
        health_status["services"]["mongodb"] = "healthy"
    except Exception as e:
        health_status["services"]["mongodb"] = "unavailable"
        logger.error(f"MongoDB health check failed: {str(e)}")
    
    try:
        redis_handler.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = "unavailable"
        logger.error(f"Redis health check failed: {str(e)}")
    
    return health_status
