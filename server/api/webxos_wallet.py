# server/api/webxos_wallet.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
from server.security.auth import oauth2_scheme
import logging
from typing import Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

class WalletTransaction(BaseModel):
    address: str
    amount: float
    action: str  # 'deposit', 'withdraw', 'stake'

class WalletConfig(BaseModel):
    address: str
    dao_proposal: Dict[str, Any] | None = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/wallet/transaction")
async def wallet_transaction(
    transaction: WalletTransaction,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """Handle wallet transactions with OAuth2.0 authentication."""
    try:
        wallet = db.query(Wallet).filter_by(address=transaction.address).first()
        if not wallet:
            raise HTTPException(status_code=404, detail="Wallet not found")
        
        if transaction.action == "deposit":
            wallet.balance += transaction.amount
        elif transaction.action == "withdraw":
            if wallet.balance < transaction.amount:
                raise HTTPException(
                    status_code=400,
                    detail="Insufficient balance"
                )
            wallet.balance -= transaction.amount
        elif transaction.action == "stake":
            wallet.staked_amount += transaction.amount
            wallet.balance -= transaction.amount
        
        db.commit()
        logger.info(
            f"Wallet {transaction.address} {transaction.action}: "
            f"{transaction.amount}"
        )
        return {
            "status": "success",
            "balance": wallet.balance,
            "staked_amount": wallet.staked_amount
        }
    except Exception as e:
        logger.error(f"Wallet transaction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/wallet/import")
async def import_wallet(
    config: WalletConfig,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """Import wallet configuration."""
    try:
        wallet = Wallet(
            address=config.address,
            balance=0.0,
            staked_amount=0.0,
            dao_proposal=config.dao_proposal
        )
        db.add(wallet)
        db.commit()
        logger.info(f"Wallet {config.address} imported")
        return {"status": "success", "address": config.address}
    except Exception as e:
        logger.error(f"Wallet import error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallet/export/{address}")
async def export_wallet(
    address: str,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """Export wallet configuration."""
    try:
        wallet = db.query(Wallet).filter_by(address=address).first()
        if not wallet:
            raise HTTPException(status_code=404, detail="Wallet not found")
        return {
            "address": wallet.address,
            "balance": wallet.balance,
            "staked_amount": wallet.staked_amount,
            "dao_proposal": wallet.dao_proposal
        }
    except Exception as e:
        logger.error(f"Wallet export error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
