from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.orm import Session
from server.config.database import Base
import time
import logging

logging.basicConfig(level=logging.INFO, filename="logs/transaction_repository.log")

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_id = Column(String, nullable=False)
    transaction_type = Column(String, nullable=False)  # e.g., "reward", "transfer"
    amount = Column(Integer, nullable=False)
    description = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)

class TransactionRepository:
    def __init__(self, db: Session):
        self.db = db

    def log_transaction(self, wallet_id: str, transaction_type: str, amount: int, description: str) -> None:
        """Log a DAO transaction."""
        try:
            transaction = Transaction(
                wallet_id=wallet_id,
                transaction_type=transaction_type,
                amount=amount,
                description=description,
                timestamp=time.time()
            )
            self.db.add(transaction)
            self.db.commit()
        except Exception as e:
            logging.error(f"Failed to log transaction: {str(e)}")
            self.db.rollback()
            raise

    def get_transactions(self, wallet_id: str, time_range: str = "1h") -> list:
        """Get transactions for a wallet."""
        try:
            time_delta = {"1h": 3600, "24h": 86400, "7d": 604800}.get(time_range, 3600)
            transactions = self.db.query(Transaction).filter(
                Transaction.wallet_id == wallet_id,
                Transaction.timestamp >= time.time() - time_delta
            ).all()
            return [
                {
                    "id": t.id,
                    "type": t.transaction_type,
                    "amount": t.amount,
                    "description": t.description,
                    "timestamp": t.timestamp
                }
                for t in transactions
            ]
        except Exception as e:
            logging.error(f"Failed to get transactions: {str(e)}")
            raise
