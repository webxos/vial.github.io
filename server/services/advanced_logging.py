import logging


logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedLogger:
    def log(self, message, extra=None):
        logger.info(message, extra=extra or {})


def setup_logging(app):
    app.state.logger = AdvancedLogger()
    app.state.logger.log("Visual config system initialized")
