import json
import os
import logging
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy import Column, String, Float, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from server.services.database import get_db
from server.services.mongodb_handler import MongoDBHandler
from server.config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Base = declarative_base()


class Wallet(Base):
    __tablename__ = "wallets"
    user_id = Column(String, primary_key=True)
    balance = Column(Float, default=0.0)
    reputation = Column(Integer, default=0)
    role = Column(String, default="user")
    transactions = Column(Text, default="[]")


class WalletModel(BaseModel):
    user_id: str
    balance: float = 0.0
    reputation: int = 0
    role: str = "user"
    transactions: list = []

    class Config:
        from_attributes = True


class WebXOSWallet:
    def __init__(self):
        self.mongo = MongoDBHandler()
        self.collection = self.mongo.db["wallet"]
        async with get_db() as db:
            Base.metadata.create_all(db.bind)

    async def update_wallet(self, user_id: str, transaction: dict, balance_increment: float = float(os.getenv("WALLET_INCREMENT", 0.0001))):
        try:
            # Fetch or initialize wallet in MongoDB
            wallet = await self.mongo.get_metadata({"user_id": user_id}) or {
                "user_id": user_id,
                "webxos": 0.0,
                "transactions": []
            }
            wallet["transactions"].append({
                **transaction,
                "timestamp": datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + balance_increment

            # Update MongoDB
            await self.collection.update_one(
                {"user_id": user_id},
                {"$set": {"webxos": wallet["webxos"], "transactions": wallet["transactions"]}},
                upsert=True
            )

            # Update SQLite
            async with get_db() as db:
                wallet_db = await db.execute(
                    select(Wallet).filter_by(user_id=user_id)
                )
                wallet_db = wallet_db.scalar_one_or_none()
                if not wallet_db:
                    wallet_db = Wallet(
                        user_id=user_id,
                        balance=wallet["webxos"],
                        transactions=json.dumps(wallet["transactions"])
                    )
                    db.add(wallet_db)
                else:
                    wallet_db.balance = wallet["webxos"]
                    wallet_db.transactions = json.dumps(wallet["transactions"])
                await db.commit()

            return WalletModel(
                user_id=user_id,
                balance=wallet["webxos"],
                transactions=wallet["transactions"]
            )
        except Exception as e:
            logger.error(f"Wallet update error: {str(e)}")
            os.makedirs("db", exist_ok=True)
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.utcnow().isoformat()}]** Wallet update error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))
