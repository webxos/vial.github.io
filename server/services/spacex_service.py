```python
import httpx
import os

class SpaceXService:
    def __init__(self):
        self.base_url = os.getenv("SPACEX_API_URL", "https://api.spacexdata.com/v4")

    async def fetch_launches(self, limit: int = 10):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/launches?limit={limit}")
            response.raise_for_status()
            return response.json()

    async def fetch_starlink(self, limit: int = 100):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/starlink?limit={limit}")
            response.raise_for_status()
            return [sat for sat in response.json() if sat.get("latitude") and sat.get("longitude")]
```
