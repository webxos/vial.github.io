from fastapi import APIRouter

router = APIRouter()

@router.get("/help")
async def help():
    return {"message": "Vial MCP Help", "docs": "https://vial.github.io/docs"}
