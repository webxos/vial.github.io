from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.services.advanced_logging import AdvancedLogger


router = APIRouter()
logger = AdvancedLogger()


@router.get("/diagnose")
async def diagnose_system(db: Session = Depends(get_db)):
    diagnostics = {}
    try:
        db.execute("SELECT 1")
        diagnostics["database"] = "connected"
    except Exception as e:
        diagnostics["database"] = f"error: {str(e)}"
    diagnostics["version"] = "2.9.3"
    logger.log("System diagnostics run", extra={"diagnostics": diagnostics})
    return {"diagnostics": diagnostics}
