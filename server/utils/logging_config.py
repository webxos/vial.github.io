import logging
from logging.handlers import RotatingFileHandler
import structlog
import os

def configure_logging():
    """Configure structured logging with file and console output."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        "vial_mcp.log", maxBytes=10_000_000, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    # Root logger
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().addHandler(file_handler)

    # Prometheus metrics logging (placeholder)
    # logger = structlog.get_logger()
    # logger.bind(prometheus_metrics=True)

configure_logging()
