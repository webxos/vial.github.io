import json
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from server.models.webxos_wallet import WalletModel
from server.services.database import get_db
from server.utils import parse_json, validate_wallet
from server.logging import logger

router = APIRouter()


@router.post("/export")
async def export_wallet(user_id: str, db: Session = Depends(get_db)):
    wallet = db.query(WalletModel).filter(WalletModel.user_id == user_id).first()
    if not wallet:
        wallet = WalletModel(user_id=user_id, balance=72017.0,
                            network_id=str(uuid.uuid4()))
        db.add(wallet)
        db.commit()
    export_data = {
        "network_id": wallet.network_id,
        "session_start": datetime.utcnow().isoformat() + "Z",
        "reputation": 1229811727985,
        "wallet": {
            "key": str(uuid.uuid4()),
            "balance": wallet.balance,
            "address": str(uuid.uuid4()),
            "hash": "042e2b6c16cc0471417e0bca0161be72258214efcf46953a63c6343b187887ce"
        },
        "vials": []
    }
    for i in range(1, 5):
        vial_data = {
            "name": f"vial{i}",
            "status": "running",
            "language": "Python",
            "balance": 18004.25,
            "address": str(uuid.uuid4()),
            "hash": "042e2b6c16cc0471417e0bca0161be72258214efcf46953a63c6343b187887ce"
        }
        export_data["vials"].append(vial_data)
    logger.log(f"Exported wallet for user: {user_id}")
    return export_data


@router.post("/import")
async def import_wallet(file_content: str, db: Session = Depends(get_db)):
    try:
        wallet_data = parse_json(file_content)
        validated_data = validate_wallet(wallet_data)
        wallet = WalletModel(user_id=validated_data["user_id"],
                            balance=validated_data["balance"],
                            network_id=validated_data["network_id"])
        db.merge(wallet)
        db.commit()
        logger.log(f"Imported wallet for user: {validated_data['user_id']}")
        return {"status": "imported", "wallet": validated_data}
    except Exception as e:
        logger.log(f"Import failed: {str(e)}")
        return {"error": str(e)}
