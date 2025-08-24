from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import Session
from server.config.database import Base
import time
import logging

logging.basicConfig(level=logging.INFO, filename="logs/audit_repository.log")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String, nullable=False)
    wallet_id = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)
    details = Column(String)

class AuditRepository:
    def __init__(self, db: Session):
        self.db = db

    def log_action(self, action: str, wallet_id: str, endpoint: str, details: str = "") -> None:
        """Log an audit action."""
        try:
            audit = AuditLog(
                action=action,
                wallet_id=wallet_id,
                endpoint=endpoint,
                timestamp=time.time(),
                details=details
            )
            self.db.add(audit)
            self.db.commit()
            logging.info(f"Logged audit: {action} for {wallet_id} on {endpoint}")
        except Exception as e:
            logging.error(f"Audit logging error: {str(e)}")
            self.db.rollback()
            raise

    def get_audit_logs(self, wallet_id: str = None, time_range: str = "1h") -> list:
        """Retrieve audit logs."""
        try:
            time_delta = {"1h": 3600, "24h": 86400, "7d": 604800}.get(time_range, 3600)
            query = self.db.query(AuditLog).filter(AuditLog.timestamp >= time.time() - time_delta)
            if wallet_id:
                query = query.filter(AuditLog.wallet_id == wallet_id)
            logs = query.all()
            return [
                {
                    "id": log.id,
                    "action": log.action,
                    "wallet_id": log.wallet_id,
                    "endpoint": log.endpoint,
                    "timestamp": log.timestamp,
                    "details": log.details
                }
                for log in logs
            ]
        except Exception as e:
            logging.error(f"Audit retrieval error: {str(e)}")
            raise
