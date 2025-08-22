import logging.handlers


class AdvancedLogger:
    def __init__(self):
        self.logger = logging.getLogger("vial_mcp")
        handler = logging.handlers.RotatingFileHandler("vial.log", maxBytes=10000)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log(self, message: str):
        self.logger.info(message)


logger = AdvancedLogger()
