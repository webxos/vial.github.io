from fastapi import APIRouter, Depends
import uuid
from datetime import datetime
from .auth import get_current_user

router = APIRouter()

class WebXOSWallet:
    def __init__(self):
        self.wallets = {}
    
    def create_wallet(self, user_id: str):
        wallet_id = str(uuid.uuid4())
        self.wallets[wallet_id] = {
            "wallet_id": wallet_id,
            "address": str(uuid.uuid4()),
            "balance": 100.0,
            "created_at": datetime.now().isoformat()
        }
        return self.wallets[wallet_id]

webxos_wallet = WebXOSWallet()

@router.post("/create")
async def create_wallet(current_user: dict = Depends(get_current_user)):
    return webxos_wallet.create_wallet(current_user["username"])
