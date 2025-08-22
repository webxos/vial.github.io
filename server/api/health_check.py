from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.services.advanced_logging import AdvancedLogger


router = APIRouter()
logger = AdvancedLogger()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        logger.log("Health check passed", extra={"status": "healthy"})
        return {"status": "healthy", "version": "2.9.3", "database": "connected"}
    except Exception as e:
        logger.log("Health check failed", extra={"error": str(e)})
        return {"status": "unhealthy", "error": str(e)}
