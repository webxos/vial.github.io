from sqlalchemy import create_engine, Column, String, Float, Integer, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from server.security.crypto_engine import CryptoEngine
import os
from pydantic import BaseModel, validator
import re

Base = declarative_base()
engine = create_engine("sqlite:///webxos.db")
Session = sessionmaker(bind=engine)

class WalletModel(BaseModel):
    address: str
    private_key: str
    balance: float = 0.0

    @validator('address')
    def validate_address(cls, v):
        if not re.match(r'^0x[a-fA-F0-9]{40}$', v):
            raise ValueError('Invalid wallet address')
        return v

    @validator('private_key')
    def validate_private_key(cls, v):
        if not re.match(r'^[a-fA-F0-9]{64}$', v):
            raise ValueError('Invalid private key')
        return v

class Wallet(Base):
    __tablename__ = "wallets"
    id = Column(Integer, primary_key=True)
    address = Column(String, unique=True, nullable=False)
    private_key = Column(String, nullable=False)  # Encrypted
    balance = Column(Float, default=0.0)

    __table_args__ = (Index('idx_wallet_address', 'address'),)

    def __init__(self, address: str, private_key: str, balance: float = 0.0):
        crypto = CryptoEngine(os.getenv("WALLET_PASSWORD", "secure_wallet_password"))
        self.address = address
        self.private_key = crypto.encrypt(private_key)
        self.balance = balance

    def decrypt_private_key(self) -> str:
        crypto = CryptoEngine(os.getenv("WALLET_PASSWORD", "secure_wallet_password"))
        return crypto.decrypt(self.private_key)

def init_db():
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    init_db()
