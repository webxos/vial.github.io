from fastapi import APIRouter, Depends
from server.security import verify_token
import psutil

router = APIRouter()


async def get_system_metrics(token: str = Depends(verify_token)):
    metrics = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage("/").percent
    }
    return {"metrics": metrics}
