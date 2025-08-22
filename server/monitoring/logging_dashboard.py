# server/monitoring/logging_dashboard.py
from fastapi import APIRouter, Response
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging
from typing import List, Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/logs")
async def get_logs() -> List[Dict[str, Any]]:
    """Retrieve logs for dashboard."""
    try:
        with SessionLocal() as session:
            wallets = session.query(Wallet).all()
            log_data = [
                {
                    "address": wallet.address,
                    "balance": wallet.balance,
                    "staked_amount": wallet.staked_amount,
                    "dao_proposal": wallet.dao_proposal
                }
                for wallet in wallets
            ]
            logger.info("Logs retrieved for dashboard")
            return log_data
    except Exception as e:
        logger.error(f"Error retrieving logs: {str(e)}")
        raise

@router.get("/logs/visual")
async def get_visual_logs():
    """Retrieve logs for visual dashboard."""
    try:
        with SessionLocal() as session:
            wallets = session.query(Wallet).all()
            svg_content = (
                '<svg width="800" height="600" '
                'xmlns="http://www.w3.org/2000/svg">'
            )
            for wallet in wallets:
                svg_content += (
                    f'<circle cx="{wallet.balance * 10}" cy="100" r="20" '
                    f'fill="blue"/><text x="{wallet.balance * 10 + 25}" '
                    f'y="105" fill="black">{wallet.address}</text>'
                )
            svg_content += '</svg>'
            return Response(
                content=svg_content,
                media_type="image/svg+xml"
            )
    except Exception as e:
        logger.error(f"Error generating visual logs: {str(e)}")
        raise
