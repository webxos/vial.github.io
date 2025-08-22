import logging


logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LogHandler:
    def log(self, message, extra=None):
        logger.info(message, extra=extra or {})
