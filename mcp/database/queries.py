from sqlalchemy.orm import Session
from .data_models import Wallet
from ..auth import get_current_user
from datetime import datetime

def get_wallet_by_id(db: Session, wallet_id: str, current_user: dict = Depends(get_current_user)):
    wallet = db.query(Wallet).filter(Wallet.wallet_id == wallet_id).first()
    if wallet:
        return {"wallet_id": wallet.wallet_id, "address": wallet.address, "time": "06:11 PM EDT, Aug 20, 2025"}
    return {"error": "Wallet not found"}
