# server/services/advanced_logging.py
from fastapi import FastAPI, Request
import logging
from server.services.reputation_logger import ReputationLogger
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AdvancedLogging:
    def __init__(self, app: FastAPI):
        self.app = app
        self.register_middleware()

    def register_middleware(self):
        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            """Log requests and reputation updates."""
            try:
                response = await call_next(request)
                wallet_address = request.headers.get("X-Wallet-Address")
                if wallet_address:
                    reputation_logger = ReputationLogger()
                    await reputation_logger.log_reputation(
                        wallet_address,
                        f"wallet_{wallet_address}.md"
                    )
                logger.info(
                    f"Request: {request.method} {request.url.path} "
                    f"from {wallet_address or request.client.host}"
                )
                return response
            except Exception as e:
                logger.error(f"Logging error: {str(e)}")
                raise
