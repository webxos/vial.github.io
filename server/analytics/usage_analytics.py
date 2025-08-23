from fastapi import FastAPI, Request
from server.models.webxos_wallet import Wallet
from server.services.database import SessionLocal
from server.logging import logger
from pymongo import MongoClient
from datetime import datetime
import os

class UsageAnalytics:
    def __init__(self, app: FastAPI):
        self.app = app
        self.mongo_client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]

    async def analytics_middleware(self, request: Request, call_next):
        start_time = datetime.now()
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        try:
            with SessionLocal() as db:
                wallet = db.query(Wallet).filter(
                    Wallet.address == request.headers.get("X-Wallet-Address")
                ).first()
                if wallet:
                    self.db.analytics.insert_one({
                        "address": wallet.address,
                        "path": request.url.path,
                        "duration": duration,
                        "timestamp": int(datetime.now().timestamp())
                    })
                    logger.log(
                        f"Wallet {wallet.address} accessed {request.url.path} "
                        f"in {duration:.2f}s",
                        request_id=str(uuid.uuid4())
                    )
        except Exception as e:
            logger.log(f"Analytics error: {str(e)}", request_id=str(uuid.uuid4()))
        return response

    async def log_component_usage(self, component_id: str, action: str):
        request_id = str(uuid.uuid4())
        try:
            self.db.analytics.insert_one({
                "component_id": component_id,
                "action": action,
                "timestamp": int(datetime.now().timestamp())
            })
            logger.log(
                f"Component {component_id} performed {action}",
                request_id=request_id
            )
        except Exception as e:
            logger.log(f"Component usage error: {str(e)}", request_id=request_id)
