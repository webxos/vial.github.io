from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
import httpx
import os
from typing import Dict
from datetime import datetime
from fastapi.responses import StreamingResponse
import io
from PIL import Image

class TelescopeService:
    def __init__(self):
        self.nasa_api_key = os.getenv("NASA_API_KEY")
        self.gibs_base_url = "https://gibs.earthdata.nasa.gov"
        self.apod_base_url = "https://api.nasa.gov/planetary/apod"

    async def stream_gibs_image(self, date: str, layer: str) -> StreamingResponse:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.gibs_base_url}/wmts/epsg4326/best/{layer}/default/{date}/250m/0/0/0.jpg",
                    params={"api_key": self.nasa_api_key}
                )
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                return StreamingResponse(img_byte_arr, media_type="image/jpeg")
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch GIBS image")

    async def get_apod(self, date: str) -> Dict:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.apod_base_url,
                    params={"api_key": self.nasa_api_key, "date": date}
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch APOD data")

telescope_service = TelescopeService()

router = APIRouter(prefix="/mcp/telescope", tags=["telescope"])

@router.get("/gibs")
async def stream_gibs(date: str = datetime.now().strftime("%Y-%m-%d"), layer: str = "MODIS_Terra_CorrectedReflectance_TrueColor", token: dict = Depends(verify_token)):
    return await telescope_service.stream_gibs_image(date, layer)

@router.get("/apod")
async def get_apod(date: str = datetime.now().strftime("%Y-%m-%d"), token: dict = Depends(verify_token)) -> Dict:
    return await telescope_service.get_apod(date)
