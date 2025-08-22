import logging
import json


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(extra)s')
logger = logging.getLogger(__name__)


class LogHandler:
    def log(self, message: str, extra: dict = None):
        extra = extra or {}
        logger.info(message, extra={'extra': json.dumps(extra)})
