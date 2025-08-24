from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.orm import Session
from server.config.database import Base
import time
import logging

logging.basicConfig(level=logging.INFO, filename="logs/analytics.log")

class LLMAnalytics(Base):
    __tablename__ = "llm_analytics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)
    request_time = Column(Float, nullable=False)
    success = Column(Integer, default=1)  # 1 for success, 0 for failure

class AnalyticsRepository:
    def __init__(self, db: Session):
        self.db = db

    def log_request(self, provider: str, latency_ms: float, success: bool) -> None:
        """Log an LLM request's performance."""
        try:
            analytics = LLMAnalytics(
                provider=provider,
                latency_ms=latency_ms,
                request_time=time.time(),
                success=1 if success else 0
            )
            self.db.add(analytics)
            self.db.commit()
        except Exception as e:
            logging.error(f"Failed to log analytics: {str(e)}")
            self.db.rollback()
            raise

    def get_metrics(self, provider: str = None, time_range: str = "1h") -> dict:
        """Get LLM performance metrics."""
        try:
            time_delta = {"1h": 3600, "24h": 86400, "7d": 604800}.get(time_range, 3600)
            query = self.db.query(LLMAnalytics).filter(LLMAnalytics.request_time >= time.time() - time_delta)
            if provider:
                query = query.filter(LLMAnalytics.provider == provider)
            records = query.all()
            total_requests = len(records)
            success_count = sum(r.success for r in records)
            avg_latency = sum(r.latency_ms for r in records) / total_requests if total_requests else 0
            return {
                "avg_latency": avg_latency,
                "error_rate": 1 - (success_count / total_requests) if total_requests else 0,
                "request_count": total_requests
            }
        except Exception as e:
            logging.error(f"Failed to get metrics: {str(e)}")
            raise
