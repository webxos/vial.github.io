import logging
from fastapi import FastAPI


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedLogger:
    def log(self, message: str, extra: dict = None):
        logger.info(message, extra=extra or {})


def setup_logging(app: FastAPI):
    app.state.logger = AdvancedLogger()
    app.state.logger.log("Visual config system initialized", extra={"system": "vial_mcp"})
