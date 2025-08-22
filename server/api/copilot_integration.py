from fastapi import APIRouter
from server.services.git_trainer import GitTrainer
from server.models.copilot_model import CopilotModel


router = APIRouter()

class CopilotIntegration:
    def __init__(self):
        self.git_trainer = GitTrainer()

    async def suggest_code(self, repo_path: str, file_path: str):
        diff = await self.git_trainer.get_diff(repo_path, file_path)
        return CopilotModel(suggestion=f"// Suggested change for {file_path}: {diff}")


copilot = CopilotIntegration()

@router.post("/suggest")
async def suggest_code_endpoint(repo_path: str, file_path: str):
    return await copilot.suggest_code(repo_path, file_path)
