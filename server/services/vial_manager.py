import os
import logging
from typing import List, Dict, Any
from pydantic import BaseModel
import torch
from pymongo import MongoClient
from redis.asyncio import Redis
from mcp import tool
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/vial_mcp"))
redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
db = mongo_client.vial_mcp

class VialAgent(BaseModel):
    id: str
    model_type: str = "pytorch"
    status: str = "idle"
    task: Optional[str] = None

@tool("vial_status")
async def get_vial_status() -> List[Dict[str, Any]]:
    """Get status of all vial agents."""
    try:
        agents = db.agents.find()
        return [VialAgent(**agent).dict() for agent in agents]
    except Exception as e:
        logger.error(f"Get vial status failed: {str(e)}")
        return []

@tool("vial_run_task")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def run_vial_task(agent_id: str, task: str) -> Dict[str, Any]:
    """Run a task on a vial agent."""
    try:
        agent = db.agents.find_one({"id": agent_id})
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Simulate PyTorch task
        model = torch.nn.Linear(10, 1)  # Placeholder
        result = {"output": model(torch.randn(1, 10)).item()}
        
        db.agents.update_one({"id": agent_id}, {"$set": {"status": "running", "task": task}})
        await redis_client.setex(f"vial:{agent_id}:result", 3600, str(result))
        return result
    except Exception as e:
        logger.error(f"Vial task failed: {str(e)}")
        raise
