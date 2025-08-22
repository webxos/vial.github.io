from fastapi import FastAPI, APIRouter
from server.services.advanced_logging import AdvancedLogger


router = APIRouter()
logger = AdvancedLogger()


def setup_logging_dashboard(app: FastAPI):
    @router.get("/logs")
    async def get_logs():
        logs = [
            {"timestamp": "2025-08-22T12:45:00", "level": "INFO", "message": "System initialized", "extra": {"system": "vial_mcp"}},
            {"timestamp": "2025-08-22T12:46:00", "level": "INFO", "message": "API request processed", "extra": {"endpoint": "/health"}}
        ]
        logger.log("Logs retrieved for dashboard", extra={"log_count": len(logs)})
        return {"logs": logs}
    
    app.include_router(router, prefix="/dashboard")
    logger.log("Logging dashboard initialized", extra={"system": "dashboard"})
