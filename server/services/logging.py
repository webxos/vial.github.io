import logging
from server.services.audit_log import AuditLog


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="/app/logs/vial.log"
)


class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.audit = AuditLog()

    async def info(self, message: str, user_id: str = "system"):
        self.logger.info(message)
        await self.audit.log_action(
            action="log_info",
            user_id=user_id,
            details={"message": message}
        )

    async def error(self, message: str, user_id: str = "system"):
        self.logger.error(message)
        await self.audit.log_action(
            action="log_error",
            user_id=user_id,
            details={"message": message}
        )
