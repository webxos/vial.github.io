from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.services.swarm_agent import SwarmAgent
from server.models.dao_repository import DAORepository
from server.models.user_repository import UserRepository
from server.config.database import get_db
from sqlalchemy.orm import Session
from fastapi.security import OAuth2AuthorizationCodeBearer
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO, filename="logs/task_orchestrator.log")

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="/mcp/auth/authorize",
    tokenUrl="/mcp/auth/token",
    scheme_name="OAuth2PKCE"
)

class TaskRequest(BaseModel):
    tasks: list[dict]
    wallet_id: str

@router.post("/orchestrate")
async def orchestrate_tasks(request: TaskRequest, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        # Validate wallet
        user_repo = UserRepository(db)
        user = user_repo.db.query(user_repo.User).filter(user_repo.User.wallet_id == request.wallet_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Wallet not found")
        
        # Validate tasks
        valid_task_types = ["visualization", "computation"]
        for task in request.tasks:
            if task.get("type") not in valid_task_types:
                raise HTTPException(status_code=400, detail=f"Invalid task type: {task.get('type')}")
        
        # Process tasks with swarm agent
        swarm_agent = SwarmAgent(db)
        results = await swarm_agent.distribute_tasks(request.tasks, request.wallet_id)
        
        # Award DAO points
        dao_repo = DAORepository(db)
        dao_repo.add_reputation(request.wallet_id, points=len(request.tasks) * 5)
        
        return {
            "results": results,
            "reputation_points": dao_repo.get_reputation(request.wallet_id)
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Task orchestration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
