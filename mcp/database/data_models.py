from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Wallet(Base):
    __tablename__ = "wallets"
    id = Column(Integer, primary_key=True, index=True)
    wallet_id = Column(String(36), unique=True, index=True)
    address = Column(String(36))
    quantum_state = Column(String)
    created_at = Column(DateTime, default=datetime.now)

class DAOProposal(Base):
    __tablename__ = "dao_proposals"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500))
    description = Column(String)
    creator = Column(String(255))
    voting_start = Column(DateTime, default=datetime.now)
    voting_end = Column(DateTime)
    status = Column(String(50))
