from fastapi import APIRouter
from server.mcp_server import app


router = APIRouter()

@router.get("/troubleshoot")
async def troubleshoot():
    return {"status": "running", "checks": ["db", "api", "network"]}
