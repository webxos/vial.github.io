import asyncio
import platform
from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
from typing import Dict, List
import random
from datetime import datetime, timedelta

class DropshipService:
    def __init__(self):
        self.planets = ["Earth", "Moon", "Mars"]
        self.mission_id = 0

    def simulate_dropship(self, origin: str, destination: str, cargo: int) -> Dict:
        if origin not in self.planets or destination not in self.planets:
            raise ValueError("Invalid origin or destination")
        self.mission_id += 1
        distance = random.uniform(0.38e6, 401e6)  # Distance in km
        travel_time = distance / 1000  # Simplified time in hours
        eta = (datetime.now() + timedelta(hours=travel_time)).strftime("%Y-%m-%d %H:%M:%S")
        return {
            "mission_id": self.mission_id,
            "origin": origin,
            "destination": destination,
            "cargo": cargo,
            "distance_km": distance,
            "eta": eta
        }

dropship_service = DropshipService()

router = APIRouter(prefix="/mcp/dropship", tags=["dropship"])

@router.get("/simulate")
async def simulate_dropship(origin: str = "Earth", destination: str = "Mars", cargo: int = 1000, token: dict = Depends(verify_token)) -> Dict:
    try:
        return dropship_service.simulate_dropship(origin, destination, cargo)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

async def main():
    mission = dropship_service.simulate_dropship("Earth", "Mars", 1000)
    print(mission)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
