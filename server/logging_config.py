import logging
import os
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - RequestID: %(request_id)s')

file_handler = logging.FileHandler('errorlog.md')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log(message: str, level: str = 'info', request_id: str = None):
    request_id = request_id or str(uuid.uuid4())
    extra = {'request_id': request_id}
    if level == 'error':
        logger.error(message, extra=extra)
    else:
        logger.info(message, extra=extra)
