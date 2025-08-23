import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from httpx import AsyncClient
from pymongo import MongoClient
from datetime import datetime

logger = logging.getLogger(__name__)
app = FastAPI()

class NASAData(BaseModel):
    dataset: str
    data: dict

@app.get("/v1/nasa/fetch")
async def fetch_nasa_data(dataset: str = "apod"):
    """Fetch data from NASA API."""
    try:
        async with AsyncClient() as client:
            shield_response = await client.post(
                "https://api.azure.ai/content-safety/prompt-shields",
                json={"prompt": dataset}
            )
            if shield_response.json().get("malicious"):
                raise HTTPException(status_code=400, detail="Malicious input detected")

            response = await client.get(
                f"https://api.nasa.gov/planetary/{dataset}",
                params={"api_key": "DEMO_KEY"}  # Use env var in production
            )
            response.raise_for_status()
            data = response.json()

            mongo_client = MongoClient("mongodb://mongo:27017")
            mongo_client.vial_mcp.research_data.insert_one({
                "type": "NASA",
                "dataset": dataset,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            })
            return {"data": data}
    except Exception as e:
        logger.error(f"NASA API fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
