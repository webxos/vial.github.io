import logging
from logging.handlers import RotatingFileHandler


class Logger:
    def __init__(self):
        self.logger = logging.getLogger("vial")
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler("vial.log", maxBytes=1000000, backupCount=5)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


    def info(self, message: str):
        self.logger.info(message)


    def error(self, message: str):
        self.logger.error(message)


logger = Logger()
