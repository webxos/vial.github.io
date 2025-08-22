from fastapi import APIRouter

router = APIRouter()


@router.post("/jsonrpc")
async def handle_jsonrpc():
    return {"jsonrpc": "2.0", "result": "success"}
