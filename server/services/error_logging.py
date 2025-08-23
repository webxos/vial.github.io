from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from server.logging_config import logger
import uuid
import datetime
import os

Base = declarative_base()

class ErrorLog(Base):
    __tablename__ = "error_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String, index=True)
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class ErrorLogger:
    def __init__(self):
        db_url = os.getenv("ERROR_LOG_DB", "sqlite:///./error_logs.db")
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def log_error(self, message: str, request_id: str) -> None:
        try:
            with self.Session() as session:
                error_log = ErrorLog(request_id=request_id, message=message)
                session.add(error_log)
                session.commit()
                logger.info(f"Logged error to SQLite: {message}", request_id=request_id)
        except Exception as e:
            logger.error(f"SQLite logging error: {str(e)}", request_id=request_id)

    def get_logs(self, request_id: str = None) -> List[Dict[str, Any]]:
        try:
            with self.Session() as session:
                query = session.query(ErrorLog)
                if request_id:
                    query = query.filter_by(request_id=request_id)
                logs = [
                    {"id": log.id, "request_id": log.request_id, "message": log.message, "timestamp": log.timestamp.isoformat()}
                    for log in query.all()
                ]
                logger.info(f"Retrieved {len(logs)} error logs", request_id=str(uuid.uuid4()))
                return logs
        except Exception as e:
            logger.error(f"Error retrieving logs: {str(e)}", request_id=str(uuid.uuid4()))
            return []
