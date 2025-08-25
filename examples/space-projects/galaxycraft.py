import asyncio
import platform
from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token
from typing import Dict
import random
import json

class GalaxyCraftService:
    def generate_galaxy(self, stars: int = 100) -> Dict:
        galaxy = {
            "stars": [
                {
                    "id": i,
                    "position": [random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100)],
                    "color": f"#{random.randint(0, 0xFFFFFF):06x}",
                    "size": random.uniform(0.1, 1.0)
                } for i in range(stars)
            ]
        }
        return galaxy

galaxy_service = GalaxyCraftService()

router = APIRouter(prefix="/mcp/galaxycraft", tags=["galaxycraft"])

@router.get("/generate")
async def generate_galaxy(stars: int = 100, token: dict = Depends(verify_token)) -> Dict:
    if stars < 10 or stars > 1000:
        raise HTTPException(status_code=400, detail="Stars must be between 10 and 1000")
    return galaxy_service.generate_galaxy(stars)

async def main():
    galaxy = galaxy_service.generate_galaxy(100)
    with open("galaxy.json", "w") as f:
        json.dump(galaxy, f, indent=2)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
