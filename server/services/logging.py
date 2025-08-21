import logging
import os
from server.services.audit_log import AuditLog
from fastapi import HTTPException


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="/app/logs/vial.log"
)


class Logger:
    def __init__(self, name: str):
        try:
            os.makedirs("/app/logs", exist_ok=True)
            self.logger = logging.getLogger(name)
            self.audit = AuditLog()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Logger init failed: {str(e)}")

    async def info(self, message: str, user_id: str = "system"):
        try:
            self.logger.info(message)
            await self.audit.log_action(
                action="log_info",
                user_id=user_id,
                details={"message": message}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Logging failed: {str(e)}")

    async def error(self, message: str, user_id: str = "system"):
        try:
            self.logger.error(message)
            await self.audit.log_action(
                action="log_error",
                user_id=user_id,
                details={"message": message}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Logging failed: {str(e)}")
