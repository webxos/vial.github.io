# server/api/websocket.py
from fastapi import APIRouter, WebSocket
from server.services.vial_manager import VialManager
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging
import json

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time updates."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            with SessionLocal() as session:
                if data.get("type") == "wallet_transaction":
                    wallet = session.query(Wallet).filter_by(
                        address=data.get("wallet_address")
                    ).first()
                    if wallet:
                        wallet.balance += data.get("amount", 0)
                        session.commit()
                        await websocket.send_json({
                            "status": "success",
                            "balance": wallet.balance,
                            "address": wallet.address
                        })
                elif data.get("type") == "vial_update":
                    vial_manager = VialManager()
                    status = await vial_manager.train_vial(
                        data.get("vial_id")
                    )
                    await websocket.send_json({
                        "status": "success",
                        "vial_id": data.get("vial_id"),
                        "training_status": status
                    })
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()
