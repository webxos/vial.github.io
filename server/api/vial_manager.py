from fastapi import APIRouter, UploadFile, HTTPException
from server.mcp_server import app
from server.utils import parse_json


router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile):
    try:
        content = await file.read()
        wallet_data = parse_json(content.decode())
        return wallet_data.dict()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid wallet file")
