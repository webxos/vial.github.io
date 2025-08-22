from fastapi import APIRouter, Depends, HTTPException
from server.models.auth_agent import AuthAgent
from server.services.git_trainer import GitTrainer
import langchain
import json


router = APIRouter()

auth_agent = AuthAgent()
git_trainer = GitTrainer()


@router.post("/comms/send")
async def send_message(message: dict, user_id: str = Depends(auth_agent.authenticate)):
    try:
        response = langchain.process_message(message, user_id)
        git_trainer.commit_changes(f"Comms update: {user_id}")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
