from fastapi import APIRouter

router = APIRouter()


@router.post("/integrate")
async def integrate_copilot():
    return {"status": "integrated"}
