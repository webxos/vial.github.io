import logging
from server.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("vial.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("vial")

def setup_logging():
    logger.info("Logging initialized")
    return logger
