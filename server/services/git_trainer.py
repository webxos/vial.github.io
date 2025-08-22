# server/services/git_trainer.py
from server.services.vial_manager import VialManager
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GitTrainer:
    def __init__(self):
        self.vial_manager = VialManager()

    async def train_from_git(self, repo_url: str, wallet_address: str) -> Dict[str, Any]:
        """Train model from Git repository with reputation check."""
        try:
            with SessionLocal() as session:
                wallet = session.query(Wallet).filter_by(
                    address=wallet_address
                ).first()
                if not wallet or wallet.reputation < 10.0:
                    raise ValueError(
                        f"Insufficient reputation for {wallet_address}"
                    )
            
            result = subprocess.run(
                ["git", "clone", repo_url],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise ValueError(f"Git clone failed: {result.stderr}")
            
            training_result = await self.vial_manager.train_vial(
                repo_url.split("/")[-1]
            )
            logger.info(
                f"Training from {repo_url} for wallet {wallet_address}"
            )
            return training_result
        except Exception as e:
            logger.error(f"Git training error: {str(e)}")
            raise
