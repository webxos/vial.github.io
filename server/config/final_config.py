from server.services.advanced_logging import AdvancedLogger


logger = AdvancedLogger()


def get_final_config():
    config = {
        "app_version": "2.9.3",
        "environment": "production",
        "sql_url": "sqlite:///vial.db",
        "mongodb_url": "mongodb://mongodb:27017/vial",
        "redis_url": "redis://redis:6379"
    }
    logger.log("Final configuration aggregated",
               extra={"config_keys": list(config.keys())})
    return config
