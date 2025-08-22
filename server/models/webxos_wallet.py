from sqlalchemy import Column, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()

class WalletModel(Base):
    __tablename__ = "wallets"

    user_id = Column(String, primary_key=True)
    balance = Column(Float, nullable=False)
    network_id = Column(String, nullable=False)

    def dict(self):
        return {"user_id": self.user_id, "balance": self.balance, "network_id": self.network_id}
