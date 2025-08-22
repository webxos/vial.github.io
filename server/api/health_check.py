from fastapi import APIRouter, Depends
from server.services.vial_manager import VialManager
from server.services.database import get_db
from server.logging import logger

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        # Check database connectivity
        db.execute("SELECT 1")
        # Check vial manager status
        vial_manager = VialManager()
        vial_status = vial_manager.get_vial_status("vial1")
        # Check MCP Alchemist status
        alchemist_status = {"status": "running"}  # Placeholder for Alchemist check
        health_status = {
            "database": "connected",
            "vial_manager": "running" if vial_status else "inactive",
            "mcp_alchemist": alchemist_status["status"],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        logger.log("Health check completed successfully")
        return health_status
    except Exception as e:
        logger.log(f"Health check error: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}
