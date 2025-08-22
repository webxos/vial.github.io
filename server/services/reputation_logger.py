# server/services/reputation_logger.py
from fastapi import Depends
from sqlalchemy.orm import Session
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
from server.security.auth import oauth2_scheme
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReputationLogger:
    def __init__(self, db: Session = Depends(SessionLocal)):
        self.db = db

    async def log_reputation(self, wallet_address: str, markdown_file: str, token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
        """Log reputation from markdown wallet file."""
        try:
            # Parse markdown file for reputation data (simplified)
            reputation = 0.0
            with open(markdown_file, 'r') as f:
                content = f.read()
                match = re.search(r'reputation:\s*(\d+\.\d+)', content)
                if match:
                    reputation = float(match.group(1))
            
            # Update wallet with reputation
            wallet = self.db.query(Wallet).filter_by(address=wallet_address).first()
            if not wallet:
                raise ValueError("Wallet not found")
            
            wallet.reputation = reputation
            self.db.commit()
            
            # Log to middleware
            logger.info(f"Reputation updated for wallet {wallet_address}: {reputation}")
            
            # Secondary security check
            from server.quantum.quantum_sync import QuantumSync
            quantum_sync = QuantumSync()
            quantum_result = await quantum_sync.sync_wallet(wallet_address)
            
            return {
                "status": "success",
                "wallet_address": wallet_address,
                "reputation": reputation,
                "quantum_state": quantum_result.get("quantum_state")
            }
        except Exception as e:
            logger.error(f"Reputation logging error: {str(e)}")
            raise

    async def get_reputation(self, wallet_address: str, token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
        """Retrieve reputation for a wallet."""
        try:
            wallet = self.db.query(Wallet).filter_by(address=wallet_address).first()
            if not wallet:
                raise ValueError("Wallet not found")
            return {
                "status": "success",
                "wallet_address": wallet_address,
                "reputation": wallet.reputation
            }
        except Exception as e:
            logger.error(f"Reputation retrieval error: {str(e)}")
            raise
