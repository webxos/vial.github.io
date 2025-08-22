import uuid
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from server.models.webxos_wallet import WalletModel
from server.services.database import get_db
from server.utils import parse_json, validate_wallet
from server.logging import logger
from pydantic import BaseModel


class DAOConfig(BaseModel):
    name: str
    tokens: int
    voting_period: int


class StakeRequest(BaseModel):
    user_id: str
    amount: float


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
        "vials": [],
        "visualization": {"nodes": [], "edges": []}
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
        export_data["visualization"]["nodes"].append({"id": f"vial{i}", "label": f"Vial {i}"})
    logger.log(f"Exported wallet for user: {user_id}", extra={"user_id": user_id})
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
        logger.log(f"Imported wallet for user: {validated_data['user_id']}",
                  extra={"user_id": validated_data["user_id"]})
        return {"status": "imported", "wallet": validated_data}
    except Exception as e:
        logger.log(f"Import failed: {str(e)}", extra={"error": str(e)})
        return {"error": str(e)}


@router.post("/dao/create")
async def create_dao(dao_config: DAOConfig, db: Session = Depends(get_db)):
    dao_id = str(uuid.uuid4())
    dao_data = {
        "id": dao_id,
        "name": dao_config.name,
        "tokens": dao_config.tokens,
        "voting_period": dao_config.voting_period,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "visualization": {"nodes": [{"id": dao_id, "label": dao_config.name}]}
    }
    logger.log(f"Created DAO: {dao_config.name}", extra={"dao_id": dao_id})
    return {"status": "created", "dao": dao_data}


@router.post("/stake")
async def stake_tokens(request: StakeRequest, db: Session = Depends(get_db)):
    wallet = db.query(WalletModel).filter(WalletModel.user_id == request.user_id).first()
    if not wallet or wallet.balance < request.amount:
        logger.log(f"Stake failed: Insufficient balance for {request.user_id}",
                  extra={"user_id": request.user_id})
        return {"error": "Insufficient balance"}
    wallet.balance -= request.amount
    db.commit()
    logger.log(f"Staked {request.amount} for user: {request.user_id}",
              extra={"user_id": request.user_id, "amount": request.amount})
    return {"status": "staked", "amount": request.amount}
