# scripts/deployment_validator.py
from server.config.final_config import settings
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging

logger = logging.getLogger(__name__)

def validate_deployment():
    """Validate deployment configuration."""
    try:
        if not settings.WEBXOS_WALLET_ADDRESS:
            logger.error("WEBXOS_WALLET_ADDRESS not set")
            return False
        
        with SessionLocal() as session:
            wallet = session.query(Wallet).filter_by(
                address=settings.WEBXOS_WALLET_ADDRESS
            ).first()
            if not wallet:
                logger.error(
                    f"Wallet {settings.WEBXOS_WALLET_ADDRESS} not found"
                )
                return False
        
        if not settings.REPUTATION_LOGGING_ENABLED:
            logger.warning("Reputation logging is disabled")
        
        logger.info("Deployment validation successful")
        return True
    except Exception as e:
        logger.error(f"Deployment validation error: {str(e)}")
        return False
