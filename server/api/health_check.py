from fastapi import APIRouter
from server.mcp_server import app


router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": "2025-08-21T21:26:00Z"}
