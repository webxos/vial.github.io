from server.services.mongodb_handler import mongodb_handler
from server.logging import logger
import datetime

class AuditLog:
    def __init__(self):
        self.collection = mongodb_handler.db["audit_logs"]

    def log_action(self, user_id: str, action: str, details: dict = None):
        try:
            log_entry = {
                "user_id": user_id,
                "action": action,
                "details": details or {},
                "timestamp": datetime.datetime.utcnow()
            }
            self.collection.insert_one(log_entry)
            logger.info(f"Audit log: {user_id} performed {action}")
            return {"status": "logged", "entry": log_entry}
        except Exception as e:
            logger.error(f"Audit log error: {str(e)}")
            raise ValueError(f"Audit log failed: {str(e)}")

audit_log = AuditLog()
