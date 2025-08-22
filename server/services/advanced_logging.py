import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedLogger:
    def log(self, message):
        logger.info(message)


def setup_logging(app):
    print("Logging setup")
