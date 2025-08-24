from sqlalchemy import Column, String, Float
from sqlalchemy.orm import Session
from server.config.database import Base
import time
import logging

logging.basicConfig(level=logging.INFO, filename="logs/user_repository.log")

class User(Base):
    __tablename__ = "users"
    wallet_id = Column(String, primary_key=True)
    public_key = Column(String, nullable=False)
    created_at = Column(Float, nullable=False)
    last_accessed = Column(Float, nullable=False)

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_wallet(self, wallet_id: str, public_key: str) -> None:
        """Create a new wallet for a user."""
        try:
            user = User(
                wallet_id=wallet_id,
                public_key=public_key,
                created_at=time.time(),
                last_accessed=time.time()
            )
            self.db.add(user)
            self.db.commit()
        except Exception as e:
            logging.error(f"Failed to create wallet: {str(e)}")
            self.db.rollback()
            raise

    def update_access_time(self, wallet_id: str) -> None:
        """Update last accessed time for a wallet."""
        try:
            user = self.db.query(User).filter(User.wallet_id == wallet_id).first()
            if user:
                user.last_accessed = time.time()
                self.db.commit()
        except Exception as e:
            logging.error(f"Failed to update access time: {str(e)}")
            self.db.rollback()
            raise

    def delete_unexported_wallets(self) -> None:
        """Delete unexported wallets older than 1 hour."""
        try:
            self.db.query(User).filter(User.last_accessed < (time.time() - 3600)).delete()
            self.db.commit()
        except Exception as e:
            logging.error(f"Wallet cleanup failed: {str(e)}")
            self.db.rollback()
            raise
