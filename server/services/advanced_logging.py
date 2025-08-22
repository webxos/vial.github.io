from server.services.logging import setup_logging
from server.services.audit_log import AuditLogger
import logging

class AdvancedLogger:
    def __init__(self):
        self.logger = setup_logging()
        self.audit_logger = AuditLogger()

    def log_action(self, action: str, details: dict):
        self.logger.info(f"{action}: {details}")
        self.audit_logger.record(action, details)

def setup_advanced_logging(app):
    logger = AdvancedLogger()
    app.state.logger = logger

    @app.middleware("http")
    async def log_requests(request, call_next):
        response = await call_next(request)
        logger.log_action("HTTP_REQUEST", {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code
        })
        return response
