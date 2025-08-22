from fastapi import APIRouter
from server.logging import logger

router = APIRouter()

@router.get("/status")
async def status():
    logger.log("Status check requested")
    return {"status": "running", "version": "2.9.3"}
