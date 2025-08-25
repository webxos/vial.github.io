import httpx
import os
from typing import Dict
from fastapi import Depends
from server.api.auth_endpoint import verify_token

class NASADataClient:
    def __init__(self):
        self.api_key = os.getenv("NASA_API_KEY")
        self.base_url = "https://api.nasa.gov"

    async def query_earthdata(self, bbox: list, temporal: str, collection: str) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/planetary/earth",
                params={"bbox": ",".join(map(str, bbox)), "temporal": temporal, "collection": collection, "api_key": self.api_key}
            )
            return response.json()

    async def fetch_gibs_imagery(self, layer: str, date: str) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml",
                params={"layer": layer, "time": date, "api_key": self.api_key}
            )
            return response.json()

nasa_client = NASADataClient()

from fastapi import APIRouter
router = APIRouter(prefix="/mcp/nasa", tags=["nasa"])

@router.get("/earthdata")
async def get_earthdata(bbox: str, temporal: str, collection: str, token: dict = Depends(verify_token)):
    bbox_list = [float(x) for x in bbox.split(",")]
    return await nasa_client.query_earthdata(bbox_list, temporal, collection)

@router.get("/gibs")
async def get_gibs_imagery(layer: str, date: str, token: dict = Depends(verify_token)):
    return await nasa_client.fetch_gibs_imagery(layer, date)
