from fastapi import APIRouter, Depends, HTTPException
from server.security import verify_token
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
router = APIRouter(prefix="/jsonrpc", tags=["jsonrpc"])

MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongo:27017/vial")
client = MongoClient(MONGO_URL)
db = client.vial

@router.post("/train")
async def train(params: dict, token: str = Depends(verify_token)):
    data = params.get("data", "")
    if not data:
        raise HTTPException(status_code=400, detail="Training data required")
    # Placeholder: Implement training logic with mcp_alchemist
    db.agents.insert_one({"hash": "placeholder", "data": data, "status": "trained"})
    return {"jsonrpc": "2.0", "result": {"status": "training initiated", "data": data}, "id": params.get("id", 1)}

@router.post("/wallet")
async def wallet(token: str = Depends(verify_token)):
    # Placeholder: Fetch wallet from SQLite
    return {"jsonrpc": "2.0", "result": {"balance": 0.0, "address": "placeholder"}, "id": 1}
