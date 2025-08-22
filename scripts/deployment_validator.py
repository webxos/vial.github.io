# scripts/deployment_validator.py
from fastapi import FastAPI
from server.config.settings import settings
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging

logger = logging.getLogger(__name__)

def validate_deployment(app: FastAPI) -> bool:
    """Validate deployment configuration and dependencies."""
    try:
        # Check database connection
        with SessionLocal() as session:
            session.query(Wallet).first()
            logger.info("Database connection validated")

        # Check WebXOS wallet configuration
        if not settings.WEBXOS_WALLET_ADDRESS:
            logger.error("WEBXOS_WALLET_ADDRESS not set")
            return False

        # Check API endpoints
        response = app.test_client().get("/health")
        if response.status_code != 200:
            logger.error("Health check failed")
            return False

        # Check Three.js frontend assets
        response = app.test_client().get("/public/js/threejs_integrations.js")
        if response.status_code != 200:
            logger.error("Three.js assets not found")
            return False

        logger.info("Deployment validation successful")
        return True
    except Exception as e:
        logger.error(f"Deployment validation failed: {str(e)}")
        return False
