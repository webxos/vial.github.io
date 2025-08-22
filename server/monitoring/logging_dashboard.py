# server/monitoring/logging_dashboard.py
from fastapi import APIRouter, Depends
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
from server.security.auth import oauth2_scheme
import logging
from typing import Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/logs")
async def get_logs(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Retrieve logs with wallet and reputation data."""
    try:
        with SessionLocal() as session:
            wallets = session.query(Wallet).all()
            logs = [
                {
                    "address": w.address,
                    "balance": w.balance,
                    "reputation": w.reputation
                }
                for w in wallets
            ]
        logger.info(
            f"Logs retrieved: {len(logs)} wallets "
            f"with reputation data"
        )
        return {"status": "success", "logs": logs}
    except Exception as e:
        logger.error(f"Log retrieval error: {str(e)}")
        raise
