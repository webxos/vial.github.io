from fastapi import APIRouter
from server.services.advanced_logging import AdvancedLogger


router = APIRouter()
logger = AdvancedLogger()


@router.get("/void")
async def void_endpoint():
    logger.log("Void endpoint called", extra={"endpoint": "void"})
    return {"status": "void", "message": "Placeholder endpoint"}
