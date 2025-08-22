from fastapi import APIRouter

router = APIRouter()


@router.get("/comms")
async def get_comms():
    return {"status": "connected"}
