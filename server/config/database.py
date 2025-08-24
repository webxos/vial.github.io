from sqlalchemy import create_engine, Column, String, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from server.config.settings import Settings
import os
import logging

logging.basicConfig(level=logging.INFO, filename="logs/database.log")
settings = Settings()
Base = declarative_base()

class Wallet(Base):
    __tablename__ = "wallets"
    wallet_id = Column(String, primary_key=True)
    public_key = Column(String, nullable=False)
    reputation = Column(Integer, default=0)
    last_accessed = Column(Float, nullable=False)

engine = create_engine(f"sqlite:///{settings.wallet_dir}/wallets.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def cleanup_unexported_wallets():
    """Delete unexported wallets to maintain lightweight runtime."""
    try:
        db = SessionLocal()
        db.query(Wallet).filter(Wallet.last_accessed < (time.time() - 3600)).delete()
        db.commit()
    except Exception as e:
        logging.error(f"Wallet cleanup failed: {str(e)}")
    finally:
        db.close()
