# server/models/webxos_wallet.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, JSON

Base = declarative_base()

class Wallet(Base):
    __tablename__ = "wallets"
    
    address = Column(String, primary_key=True)
    balance = Column(Float, default=0.0)
    staked_amount = Column(Float, default=0.0)
    reputation = Column(Float, default=0.0)  # Reputation for user metrics
    dao_proposal = Column(JSON, nullable=True)  # DAO governance data
    quantum_state = Column(String, nullable=True)  # Quantum sync state
