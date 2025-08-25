from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
import requests
import torch
import os
from typing import Dict

class NASASatelliteProcessor:
    def __init__(self):
        self.api_url = "https://api.nasa.gov/neo/rest/v1/feed"
        self.api_key = os.getenv("NASA_API_KEY")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def fetch_satellite_data(self) -> Dict:
        try:
            params = {"api_key": self.api_key, "start_date": "2025-08-25", "end_date": "2025-08-26"}
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch satellite data: {str(e)}")

    async def process_trajectory(self, data: Dict) -> Dict:
        try:
            neo_data = data.get("near_earth_objects", {}).get("2025-08-25", [])
            positions = torch.tensor([[item["close_approach_data"][0]["miss_distance"]["kilometers"] for item in neo_data]], device=self.device).float()
            return {"trajectory": positions.cpu().numpy().tolist()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Trajectory processing failed: {str(e)}")

nasa_satellite = NASASatelliteProcessor()

router = APIRouter(prefix="/mcp/nasa_satellite", tags=["nasa_satellite"])

@router.get("/data")
async def get_satellite_data(token: dict = Depends(verify_token)) -> Dict:
    data = await nasa_satellite.fetch_satellite_data()
    return await nasa_satellite.process_trajectory(data)
