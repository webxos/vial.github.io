from fastapi import APIRouter
import psutil
from datetime import datetime

router = APIRouter()

@router.get("/metrics")
async def get_metrics():
    return {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage("/").percent,
        "timestamp": datetime.now().isoformat()
    }
