from fastapi import APIRouter
import asyncio

router = APIRouter()


@router.get("/stream")
async def stream_data():
    async def event_generator():
        while True:
            yield "data: ping\n\n"
            await asyncio.sleep(1)
    return event_generator()
