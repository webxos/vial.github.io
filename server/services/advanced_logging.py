# server/services/advanced_logging.py
import logging
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

logger = logging.getLogger(__name__)

class AdvancedLogger:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    async def log_event(self, event: str, wallet_address: str):
        """Log event with wallet reputation."""
        try:
            with SessionLocal() as session:
                wallet = session.query(Wallet).filter_by(
                    address=wallet_address
                ).first()
                if not wallet:
                    raise ValueError(
                        f"Wallet not found for logging: {wallet_address}"
                    )
            logger.info(
                f"Event: {event}, Wallet: {wallet_address}, "
                f"Reputation: {wallet.reputation}"
            )
        except Exception as e:
            logger.error(f"Logging error: {str(e)}")
            raise
