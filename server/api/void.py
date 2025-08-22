from fastapi import APIRouter

router = APIRouter()


@router.post("/void")
async def void_operation():
    return {"status": "voided"}
