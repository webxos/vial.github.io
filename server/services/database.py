from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

Base = declarative_base()

# SQLite User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)

# SQLite Wallet model
class Wallet(Base):
    __tablename__ = "wallets"
    id = Column(Integer, primary_key=True)
    address = Column(String, unique=True, nullable=False)
    balance = Column(Float, default=0.0)
    hash = Column(String, nullable=False)

def init_db():
    # SQLite setup
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///vial.db")
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    
    # MongoDB setup
    MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongo:27017/vial")
    client = MongoClient(MONGO_URL)
    db = client.vial
    db.agents.create_index("hash", unique=True)
    
    # Create default admin user if not exists
    Session = sessionmaker(bind=engine)
    session = Session()
    from server.security import get_password_hash
    try:
        if not session.query(User).filter_by(username="admin").first():
            admin = User(username="admin", hashed_password=get_password_hash("admin"))
            session.add(admin)
            session.commit()
    finally:
        session.close()

if __name__ == "__main__":
    init_db()
