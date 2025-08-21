from server.services.mongodb_handler import MongoDBHandler
from datetime import datetime


class AuditLog:
    def __init__(self):
        self.mongo = MongoDBHandler()
        self.collection = self.mongo.db["audit_logs"]

    async def log_action(self, action: str, user_id: str, details: dict):
        log_entry = {
            "action": action,
            "user_id": user_id,
            "details": details,
            "timestamp": datetime.utcnow()
        }
        result = await self.mongo.save_metadata(log_entry)
        return result
