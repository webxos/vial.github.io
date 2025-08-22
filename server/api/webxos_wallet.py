from fastapi import APIRouter
from server.mcp_server import app
from server.utils import parse_json


router = APIRouter()

@router.post("/export")
async def export_wallet(user_id: str):
    return {"user_id": user_id, "balance": 100.0, "network_id": "test_net"}

@router.post("/import")
async def import_wallet(data: str):
    wallet = parse_json(data)
    return wallet.dict()
