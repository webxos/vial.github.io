from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.models.webxos_wallet import WalletModel
from server.mcp.auth import oauth2_scheme
from server.logging import logger
import uuid

router = APIRouter()

class WalletUpdate(BaseModel):
    vial_id: str
    balance: float
    active: bool

@router.post("/wallet/update")
async def update_wallet(update: WalletUpdate, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        wallet = db.query(WalletModel).filter(WalletModel.vial_id == update.vial_id).first()
        if not wallet:
            raise HTTPException(status_code=404, detail="Wallet not found")
        wallet.balance = update.balance
        wallet.active = update.active
        db.commit()
        logger.log(f"Wallet updated for vial {update.vial_id}", request_id=request_id)
        return {"status": "success", "request_id": request_id}
    except Exception as e:
        logger.log(f"Wallet update error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallet/{vial_id}")
async def get_wallet(vial_id: str, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        wallet = db.query(WalletModel).filter(WalletModel.vial_id == vial_id).first()
        if not wallet:
            raise HTTPException(status_code=404, detail="Wallet not found")
        logger.log(f"Wallet retrieved for vial {vial_id}", request_id=request_id)
        return {
            "vial_id": wallet.vial_id,
            "balance": wallet.balance,
            "active": wallet.active,
            "request_id": request_id
        }
    except Exception as e:
        logger.log(f"Wallet retrieval error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/wallet/reward")
async def reward_wallet(vial_id: str, amount: float, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        wallet = db.query(WalletModel).filter(WalletModel.vial_id == vial_id).first()
        if not wallet:
            raise HTTPException(status_code=404, detail="Wallet not found")
        wallet.balance += amount
        db.commit()
        logger.log(f"Reward of {amount} WebXOS added to vial {vial_id}", request_id=request_id)
        return {"status": "success", "new_balance": wallet.balance, "request_id": request_id}
    except Exception as e:
        logger.log(f"Wallet reward error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
