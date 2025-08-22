from fastapi import APIRouter
from server.logging import logger

router = APIRouter()

@router.get("/troubleshoot")
async def troubleshoot():
    logger.log("Troubleshooting initiated")
    return {"status": "running", "checks": ["db", "api", "network"]}
