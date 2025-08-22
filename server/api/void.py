from fastapi import APIRouter
from server.mcp_server import app


router = APIRouter()

@router.post("/void/reset")
async def reset_network():
    return {"status": "network reset"}
