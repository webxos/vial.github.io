from fastapi import APIRouter, WebSocket, Depends
from server.models.webxos_wallet import WebXOSWallet
from server.api.vial_manager import VialManager
from server.services.security import get_current_user
import json
import asyncio

router = APIRouter()

@router.websocket("/stream")
async def stream_vial_states(websocket: WebSocket, user: str = Depends(get_current_user)):
    await websocket.accept()
    vial_manager = VialManager()
    wallet = WebXOSWallet()
    try:
        while True:
            states = await vial_manager.get_vial_states()
            for vial_id, state in states.items():
                transaction = {"type": "stream_update", "vial_id": vial_id, "amount": 0.0001}
                wallet_data = await wallet.update_wallet(user, transaction)
                await websocket.send_json({
                    "vial_id": vial_id,
                    "state": state,
                    "wallet": wallet_data.dict()
                })
            await asyncio.sleep(1)  # Stream every second
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
