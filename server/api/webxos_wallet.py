from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.services.advanced_logging import AdvancedLogger
from pydantic import BaseModel
import hashlib


router = APIRouter()
logger = AdvancedLogger()


class WalletTransaction(BaseModel):
    address: str
    amount: float
    hash: str


@router.post("/wallet/transfer")
async def transfer_funds(transaction: WalletTransaction, db: Session = Depends(get_db)):
    try:
        wallet_hash = hashlib.sha256(transaction.address.encode()).hexdigest()
        if wallet_hash != transaction.hash:
            logger.log("Wallet transfer failed",
                       extra={"error": "Invalid hash"})
            return {"error": "Invalid hash"}
        
        logger.log("Wallet transfer processed",
                   extra={"address": transaction.address,
                          "amount": transaction.amount})
        return {"status": "transferred", "balance": 75978.0}
    except Exception as e:
        logger.log("Wallet transfer error",
                   extra={"error": str(e)})
        raise
