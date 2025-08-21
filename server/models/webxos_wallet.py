from pydantic import BaseModel
from server.services.database import get_db
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Wallet(Base):
    __tablename__ = "wallets"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True)
    balance = Column(Float, default=0.0)
    reputation = Column(Integer, default=0)
    role = Column(String, default="user")


class WalletModel(BaseModel):
    user_id: str
    balance: float = 0.0
    reputation: int = 0
    role: str = "user"

    class Config:
        orm_mode = True


async def update_wallet(user_id: str, balance: float, reputation: int):
    async with get_db() as db:
        wallet = await db.get(Wallet, user_id)
        if not wallet:
            wallet = Wallet(user_id=user_id, balance=balance, reputation=reputation)
            db.add(wallet)
        else:
            wallet.balance = balance
            wallet.reputation = reputation
        await db.commit()
        return WalletModel.from_orm(wallet)
