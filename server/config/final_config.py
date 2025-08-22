from server.config.settings import settings
from server.config.config import load_config
from server.services.advanced_logging import AdvancedLogger


logger = AdvancedLogger()


def get_final_config():
    config = load_config()
    final_config = {
        "app_version": "2.9.3",
        "environment": "production",
        **config
    }
    logger.log("Final configuration aggregated", extra={"config_keys": list(final_config.keys())})
    return final_config
