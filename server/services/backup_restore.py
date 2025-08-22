# server/services/backup_restore.py
from fastapi import Depends, HTTPException
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def backup_data(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Backup wallet and reputation data."""
    try:
        wallets = db.query(Wallet).all()
        backup_data = [
            {
                "address": w.address,
                "balance": w.balance,
                "staked_amount": w.staked_amount,
                "reputation": w.reputation
            }
            for w in wallets
        ]
        logger.info("Backup completed")
        return {"status": "success", "data": backup_data}
    except Exception as e:
        logger.error(f"Backup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def restore_data(data: Dict[str, Any], db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Restore wallet and reputation data."""
    try:
        for item in data.get("data", []):
            wallet = Wallet(
                address=item["address"],
                balance=item["balance"],
                staked_amount=item["staked_amount"],
                reputation=item["reputation"]
            )
            db.merge(wallet)
        db.commit()
        logger.info("Restore completed")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Restore error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
