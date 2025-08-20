from fastapi import APIRouter
import psutil

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy" if psutil.cpu_percent() < 90 and psutil.virtual_memory().percent < 90 else "unhealthy",
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "time": "05:46 PM EDT, Aug 20, 2025"
    }
