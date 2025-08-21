from fastapi import APIRouter, Depends
from server.models.webxos_wallet import update_wallet, WalletModel
from server.services.security import verify_jwt


router = APIRouter()


@router.post("/export")
async def export_wallet(user_id: str, token: str = Depends(verify_jwt)):
    wallet = await update_wallet(user_id, balance=0.0, reputation=0)
    return {"status": "exported", "wallet": wallet.dict()}


@router.post("/import")
async def import_wallet(wallet: WalletModel, token: str = Depends(verify_jwt)):
    result = await update_wallet(wallet.user_id, wallet.balance, wallet.reputation)
    return {"status": "imported", "wallet": result.dict()}
