from fastapi import APIRouter

router = APIRouter()


@router.get("/help")
async def get_help():
    return {"help": "Use /auth for login"}
