# server/api/stream.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from server.services.vial_manager import VialManager
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging
import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/stream/wallet/{address}")
async def stream_wallet_updates(address: str):
    """Stream real-time wallet updates."""
    async def generate():
        with SessionLocal() as session:
            while True:
                wallet = session.query(Wallet).filter_by(address=address).first()
                if wallet:
                    yield f"data: {{\"balance\": {wallet.balance}, \"address\": \"{wallet.address}\"}}\n\n"
                await asyncio.sleep(1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@router.get("/stream/vial/{vial_id}")
async def stream_vial_training(vial_id: str):
    """Stream vial training logs."""
    async def generate():
        vial_manager = VialManager()
        for _ in range(5):  # Simulate 5 training steps
            status = await vial_manager.train_vial(vial_id)
            yield f"data: {status}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")
