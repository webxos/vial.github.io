from server.config.settings import settings
from server.services.advanced_logging import AdvancedLogger


def load_config():
    logger = AdvancedLogger()
    config = {
        "sql_url": settings.sql_url,
        "mongodb_url": settings.mongodb_url,
        "redis_url": settings.redis_url,
        "jwt_secret": settings.jwt_secret,
        "jwt_expire_minutes": settings.jwt_expire_minutes
    }
    logger.log("Configuration loaded", extra={"config_keys": list(config.keys())})
    return config
