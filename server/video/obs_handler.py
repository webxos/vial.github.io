from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from httpx import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import os

logger = logging.getLogger(__name__)
router = APIRouter()

class SVGRenderRequest(BaseModel):
    svg_data: str
    output_format: str = "mp4"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def connect_obs():
    """Connect to OBS WebSocket."""
    client = AsyncClient()
    await client.get(f"{os.getenv('OBS_WEBSOCKET_URL', 'ws://localhost:4455')}/connect")
    return client

@router.post("/obs/stream")
async def stream_svg(request: SVGRenderRequest):
    """Render SVG to video stream via OBS WebSocket."""
    try:
        client = await connect_obs()
        # Send SVG data to OBS for rendering
        response = await client.post(
            f"{os.getenv('OBS_WEBSOCKET_URL', 'ws://localhost:4455')}/render",
            json={"svg": request.svg_data, "format": request.output_format}
        )
        response.raise_for_status()
        logger.info(f"SVG streamed to OBS: {request.output_format}")
        return {"status": "success", "output": response.json()}
    except Exception as e:
        logger.error(f"OBS streaming failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
