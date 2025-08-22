# server/api/comms_hub.py
from fastapi import APIRouter, WebSocket
from server.services.vial_manager import VialManager
from server.models.webxos_wallet import Wallet
from server.services.database import SessionLocal
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            with SessionLocal() as session:
                if data.get("type") == "wallet_update":
                    wallet = session.query(Wallet).filter_by(
                        address=data.get("wallet_address")
                    ).first()
                    if wallet:
                        await websocket.send_json({
                            "status": "success",
                            "balance": wallet.balance,
                            "address": wallet.address
                        })
                elif data.get("type") == "component_update":
                    await websocket.send_json({
                        "status": "success",
                        "component_id": data.get("component_id"),
                        "action": data.get("action")
                    })
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()
