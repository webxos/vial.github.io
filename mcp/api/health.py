from fastapi import APIRouter
from ..monitoring.health import health_check

router = APIRouter()

@router.get("/health")
async def api_health_check():
    return health_check()
