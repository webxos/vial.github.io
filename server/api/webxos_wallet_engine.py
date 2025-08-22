# server/api/webxos_wallet_engine.py
from sqlalchemy.orm import Session
from server.models.webxos_wallet import Wallet
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class WalletEngine:
    def __init__(self, db: Session):
        self.db = db

    async def process_transaction(
        self,
        address: str,
        amount: float,
        action: str
    ) -> Dict[str, Any]:
        """Process wallet transactions."""
        try:
            wallet = self.db.query(Wallet).filter_by(
                address=address
            ).first()
            if not wallet:
                raise ValueError("Wallet not found")
            
            if action == "stake":
                wallet.staked_amount += amount
                wallet.balance -= amount
            elif action == "unstake":
                wallet.staked_amount -= amount
                wallet.balance += amount
            
            self.db.commit()
            logger.info(
                f"Processed {action} for wallet {address}: {amount}"
            )
            return {
                "status": "success",
                "balance": wallet.balance,
                "staked_amount": wallet.staked_amount
            }
        except Exception as e:
            logger.error(f"Transaction error: {str(e)}")
            raise

    async def process_dao_proposal(
        self,
        address: str,
        proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process DAO proposal for wallet."""
        try:
            wallet = self.db.query(Wallet).filter_by(
                address=address
            ).first()
            if not wallet:
                raise ValueError("Wallet not found")
            
            wallet.dao_proposal = proposal
            self.db.commit()
            logger.info(f"DAO proposal updated for wallet {address}")
            return {"status": "success", "proposal": proposal}
        except Exception as e:
            logger.error(f"DAO proposal error: {str(e)}")
            raise
