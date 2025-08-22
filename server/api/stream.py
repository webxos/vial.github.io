from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.get("/stream")
async def stream_data():
    async def data_stream():
        for i in range(5):
            yield f"Data chunk {i}\n"
            await asyncio.sleep(1)
    return StreamingResponse(data_stream(), media_type="text/plain")
