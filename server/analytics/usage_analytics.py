# server/analytics/usage_analytics.py
from fastapi import FastAPI, Request
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class UsageAnalytics:
    def __init__(self, app: FastAPI):
        self.app = app
        self.register_middleware()

    def register_middleware(self):
        @self.app.middleware("http")
        async def analytics_middleware(request: Request, call_next):
            start_time = datetime.now()
            response = await call_next(request)
            duration = (datetime.now() - start_time).total_seconds()

            with SessionLocal() as session:
                # Log wallet-related requests
                if "/wallet" in request.url.path:
                    wallet = session.query(Wallet).filter_by(
                        address=request.headers.get("X-Wallet-Address")
                    ).first()
                    if wallet:
                        logger.info(
                            f"Wallet {wallet.address} accessed "
                            f"{request.url.path} in {duration:.2f}s"
                        )

            return response

    async def log_component_usage(self, component_id: str, action: str):
        """Log usage of visual API router components."""
        logger.info(f"Component {component_id} performed {action}")
