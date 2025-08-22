from fastapi import APIRouter
from server.logging import logger
from server.config.settings import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    logger.log("Health check requested")
    return {"status": "ok", "timestamp": "2025-08-21T23:19:00Z", "port": settings.PORT}
