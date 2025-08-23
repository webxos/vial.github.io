# server/services/training_scheduler.py
from server.services.vial_manager import VialManager
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging

logger = logging.getLogger(__name__)

class TrainingScheduler:
    def __init__(self):
        self.vial_manager = VialManager()

    async def schedule_training(self, vial_id: str, wallet_address: str) -> dict:
        """Schedule training with reputation-based prioritization."""
        try:
            with SessionLocal() as session:
                wallet = session.query(Wallet).filter_by(
                    address=wallet_address
                ).first()
                if not wallet or wallet.reputation < 10.0:
                    raise ValueError(
                        f"Insufficient reputation for {wallet_address}"
                    )
            
            result = await self.vial_manager.train_vial(vial_id)
            logger.info(
                f"Training scheduled for vial {vial_id} "
                f"for wallet {wallet_address}"
            )
            return result
        except Exception as e:
            logger.error(f"Training scheduling error: {str(e)}")
            raise
