import httpx
from typing import Dict, List
from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token

class SpaceXService:
    BASE_URL = "https://api.spacexdata.com/v4"

    async def get_launches(self, limit: int = 10) -> List[Dict]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/launches", params={"limit": limit})
            response.raise_for_status()
            return response.json()

    async def get_starlink_satellites(self) -> List[Dict]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/starlink")
            response.raise_for_status()
            return response.json()

spacex_service = SpaceXService()

router = APIRouter(prefix="/mcp/spacex", tags=["spacex"])

@router.get("/launches")
async def get_launches(limit: int = 10, token: dict = Depends(verify_token)) -> List[Dict]:
    return await spacex_service.get_launches(limit)

@router.get("/starlink")
async def get_starlink(token: dict = Depends(verify_token)) -> List[Dict]:
    return await spacex_service.get_starlink_satellites()
