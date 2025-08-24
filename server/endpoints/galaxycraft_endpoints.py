from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from server.services.pydantic_agent import PydanticAgent
from server.services.webxos_wallet import WebXOSWallet
from sqlalchemy.orm import Session
from server.config.database import get_db
import httpx
import logging

logging.basicConfig(level=logging.INFO, filename="logs/galaxycraft.log")
router = APIRouter(prefix="/mcp/galaxycraft", tags=["Galaxycraft"])

class TrainRequest(BaseModel):
    type: str
    data: dict
    wallet_id: str

@router.post("/train")
async def train_agent(request: TrainRequest, db: Session = Depends(get_db)):
    """Train an agent for Galaxycraft tasks."""
    try:
        agent = PydanticAgent(db)
        result = await agent.execute_tool(request.type, request.data, request.wallet_id)
        
        # Calculate WebXOS reward
        wallet = WebXOSWallet(db)
        reward = await wallet.calculate_reward(request.wallet_id, {
            "complexity": 1.0,
            "quality": 0.8,
            "impact": 0.5,
            "video_source": request.data.get("video_source", "unknown")
        })
        
        logging.info(f"Agent trained for {request.type}, reward: {reward}")
        return {"status": "success", "result": result, "reward": reward}
    except Exception as e:
        logging.error(f"Error training agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nasa_data")
async def fetch_nasa_data(db: Session = Depends(get_db)):
    """Fetch NASA data for simulations."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.nasa.gov/planetary/apod", params={"api_key": "DEMO_KEY"})
            response.raise_for_status()
            data = response.json()
            logging.info(f"Fetched NASA APOD: {data['title']}")
            return data
    except Exception as e:
        logging.error(f"Error fetching NASA data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
