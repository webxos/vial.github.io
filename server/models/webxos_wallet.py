from sqlalchemy import Column, String, Float
from sqlalchemy.orm import Session
from server.config.database import Base
from server.models.user_repository import UserRepository
from server.models.dao_repository import DAOReputation
from server.models.transaction_repository import TransactionRepository
import time
import logging

logging.basicConfig(level=logging.INFO, filename="logs/webxos_wallet.log")

class WebXOSWallet(Base):
    __tablename__ = "webxos_wallets"
    wallet_id = Column(String, primary_key=True)
    public_key = Column(String, nullable=False)
    created_at = Column(Float, nullable=False)
    last_accessed = Column(Float, nullable=False)
    reputation_points = Column(Integer, default=0)

class WebXOSWalletManager:
    def __init__(self, db: Session):
        self.db = db
        self.user_repo = UserRepository(db)
        self.transaction_repo = TransactionRepository(db)

    def create_wallet(self, wallet_id: str, public_key: str) -> None:
        """Create a new WebXOS wallet."""
        try:
            wallet = WebXOSWallet(
                wallet_id=wallet_id,
                public_key=public_key,
                created_at=time.time(),
                last_accessed=time.time(),
                reputation_points=0
            )
            self.db.add(wallet)
            self.user_repo.create_wallet(wallet_id, public_key)
            self.db.commit()
            logging.info(f"Created wallet: {wallet_id}")
        except Exception as e:
            logging.error(f"Wallet creation error: {str(e)}")
            self.db.rollback()
            raise

    def update_reputation(self, wallet_id: str, points: int) -> None:
        """Update wallet reputation and log transaction."""
        try:
            wallet = self.db.query(WebXOSWallet).filter(WebXOSWallet.wallet_id == wallet_id).first()
            if not wallet:
                raise ValueError("Wallet not found")
            wallet.reputation_points += points
            wallet.last_accessed = time.time()
            self.transaction_repo.log_transaction(
                wallet_id=wallet_id,
                transaction_type="reputation_update",
                amount=points,
                description="Reputation points update"
            )
            self.db.commit()
            logging.info(f"Updated reputation for {wallet_id}: {points} points")
        except Exception as e:
            logging.error(f"Reputation update error: {str(e)}")
            self.db.rollback()
            raise

    def export_md_wallet(self, wallet_id: str) -> str:
        """Export wallet as markdown."""
        try:
            wallet = self.db.query(WebXOSWallet).filter(WebXOSWallet.wallet_id == wallet_id).first()
            if not wallet:
                raise ValueError("Wallet not found")
            md_content = f"""# WebXOS Wallet
- **Wallet ID**: {wallet.wallet_id}
- **Public Key**: {wallet.public_key}
- **Reputation Points**: {wallet.reputation_points}
- **Created At**: {time.ctime(wallet.created_at)}
- **Last Accessed**: {time.ctime(wallet.last_accessed)}
"""
            return md_content
        except Exception as e:
            logging.error(f"Wallet export error: {str(e)}")
            raise ValueError(f"Wallet export failed: {str(e)}")
