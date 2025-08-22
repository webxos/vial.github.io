from fastapi import APIRouter, Depends
from server.services.git_trainer import GitTrainer
from git import Repo
import asyncio

router = APIRouter()

class CopilotIntegration:
    def __init__(self):
        self.git_trainer = GitTrainer()

    async def suggest_code(self, repo_path: str, file_path: str):
        repo = Repo(repo_path)
        changes = await self.git_trainer.get_diff(repo, file_path)
        # Simplified copilot suggestion logic
        suggestion = f"// Suggested change for {file_path}: {changes}"
        return {"suggestion": suggestion}

copilot = CopilotIntegration()

@router.post("/copilot/suggest")
async def suggest_code_endpoint(repo_path: str, file_path: str):
    return await copilot.suggest_code(repo_path, file_path)
