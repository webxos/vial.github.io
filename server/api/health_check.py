from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from server.services.database import get_db
from server.logging import logger
import uuid

router = APIRouter()

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    request_id = str(uuid.uuid4())
    try:
        # Check database connectivity
        db.execute("SELECT 1")
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        logger.log("Health check passed", request_id=request_id)
        return status
    except Exception as e:
        logger.log(f"Health check failed: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
