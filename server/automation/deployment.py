# server/automation/deployment.py
from fastapi import FastAPI
from server.config.settings import settings
from server.services.database import SessionLocal
from server.models.visual_components import ComponentModel
from server.models.webxos_wallet import Wallet
import logging
import subprocess

logger = logging.getLogger(__name__)

def deploy_application(app: FastAPI) -> bool:
    """Deploy application with validation."""
    try:
        # Validate database
        with SessionLocal() as session:
            session.query(Wallet).first()
            components = session.query(ComponentModel).all()
            if not components:
                logger.error("No visual components found")
                return False
        
        # Validate WebXOS wallet configuration
        if not settings.WEBXOS_WALLET_ADDRESS:
            logger.error("WEBXOS_WALLET_ADDRESS not set")
            return False
        
        # Run deployment script
        result = subprocess.run(
            ["bash", "./scripts/deploy.sh"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(
                f"Deployment script failed: {result.stderr}"
            )
            return False
        
        # Validate API health
        response = app.test_client().get("/health")
        if response.status_code != 200:
            logger.error("Health check failed")
            return False
        
        logger.info("Deployment successful")
        return True
    except Exception as e:
        logger.error(f"Deployment error: {str(e)}")
        return False
