from fastapi import APIRouter, Depends
from server.security import verify_token
from server.sdk.vial_sdk import vial_sdk
import psutil
import docker

router = APIRouter(prefix="/jsonrpc", tags=["monitor"])

@router.post("/monitor")
async def monitor_system(token: str = Depends(verify_token)):
    # System metrics
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Docker status
    client = docker.from_env()
    containers = client.containers.list(all=True)
    container_status = [
        {"name": c.name, "status": c.status} for c in containers
    ]
    
    return {
        "jsonrpc": "2.0",
        "result": {
            "cpu_usage": cpu_usage,
            "memory_usage": memory.percent,
            "disk_usage": disk.percent,
            "containers": container_status
        },
        "id": 1
    }
