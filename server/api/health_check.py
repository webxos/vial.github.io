from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from server.services.database import get_db
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid


router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    request_id = str(uuid.uuid4())
    try:
        alchemist = Alchemist()
        vial_status = await alchemist.get_vial_status("vial1")
        db_status = db.execute("SELECT 1").fetchone() is not None
        health_data = {
            "status": "healthy" if db_status else "unhealthy",
            "vial_status": vial_status,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
        logger.log("Health check completed", request_id=request_id)
        return health_data
    except Exception as e:
        logger.log(f"Health check error: {str(e)}", request_id=request_id)
        return {"status": "unhealthy", "error": str(e), "request_id": request_id}
