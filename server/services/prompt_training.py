# server/services/prompt_training.py
from server.services.vial_manager import VialManager
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PromptTrainer:
    def __init__(self):
        self.vial_manager = VialManager()

    async def train_prompt(self, prompt: str, wallet_address: str) -> Dict[str, Any]:
        """Train model with prompt and reputation check."""
        try:
            with SessionLocal() as session:
                wallet = session.query(Wallet).filter_by(
                    address=wallet_address
                ).first()
                if not wallet or wallet.reputation < 5.0:
                    raise ValueError(
                        f"Insufficient reputation for {wallet_address}"
                    )
            
            result = await self.vial_manager.train_vial(prompt)
            logger.info(
                f"Prompt training completed for wallet {wallet_address}"
            )
            return result
        except Exception as e:
            logger.error(f"Prompt training error: {str(e)}")
            raise
