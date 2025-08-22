import logging
import logging.handlers
from server.config import config

class AdvancedLogger:
    def __init__(self):
        self.logger = logging.getLogger("vial_mcp")
        handler = logging.handlers.RotatingFileHandler("vial.log", maxBytes=10000, backupCount=5)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if not config.DEBUG else logging.DEBUG)

    def log(self, message: str):
        self.logger.info(message)

logger = AdvancedLogger()
