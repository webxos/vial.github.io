from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
import httpx
import os
from typing import Dict
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import numpy as np

class VideoService:
    def __init__(self):
        self.nasa_api_key = os.getenv("NASA_API_KEY")
        self.base_url = "https://gibs.earthdata.nasa.gov"

    async def stream_nasa_gibs(self, date: str, layer: str) -> StreamingResponse:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/wmts/epsg4326/best/{layer}/default/{date}/250m/0/0/0.jpg",
                    params={"api_key": self.nasa_api_key}
                )
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                return StreamingResponse(img_byte_arr, media_type="image/jpeg")
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch GIBS data")

video_service = VideoService()

router = APIRouter(prefix="/mcp/video", tags=["video"])

@router.get("/nasa-gibs")
async def stream_gibs(date: str, layer: str = "MODIS_Terra_CorrectedReflectance_TrueColor", token: dict = Depends(verify_token)):
    return await video_service.stream_nasa_gibs(date, layer)
