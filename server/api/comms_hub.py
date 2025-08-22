from fastapi import APIRouter

router = APIRouter()

@router.post("/comms/send")
async def send_message(message: str):
    return {"status": "message sent", "content": message}
