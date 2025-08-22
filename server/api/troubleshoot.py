from fastapi import APIRouter

router = APIRouter()


@router.get("/troubleshoot")
async def troubleshoot():
    return {"status": "ok"}
