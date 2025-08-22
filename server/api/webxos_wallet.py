from fastapi import APIRouter
from server.models.webxos_wallet import WalletModel
from server.utils import parse_json, validate_wallet
from server.logging import logger

router = APIRouter()

@router.post("/export")
async def export_wallet(user_id: str):
    logger.log(f"Exporting wallet for user: {user_id}")
    wallet = WalletModel(user_id=user_id, balance=100.0, network_id="test_net")
    return wallet.dict()

@router.post("/import")
async def import_wallet(data: str):
    wallet_data = parse_json(data)
    validated_data = validate_wallet(wallet_data)
    logger.log(f"Imported wallet for user: {validated_data['user_id']}")
    return validated_data
