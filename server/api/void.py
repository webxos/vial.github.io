from fastapi import APIRouter
from server.logging import logger

router = APIRouter()

@router.post("/reset")
async def reset_network():
    logger.log("Network reset initiated")
    return {"status": "network reset"}
