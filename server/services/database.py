from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator
from server.logging_config import logger
import os
import uuid

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vial_mcp.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session() -> Generator:
    request_id = str(uuid.uuid4())
    session = SessionLocal()
    try:
        logger.info("Database session opened", request_id=request_id)
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {str(e)}", request_id=request_id)
        raise
    finally:
        session.close()
        logger.info("Database session closed", request_id=request_id)
