from fastapi import APIRouter
from server.services.git_trainer import GitTrainer

router = APIRouter()

git_trainer = GitTrainer()

@router.post("/suggest")
async def suggest_code(repo_path: str, file_path: str):
    diff = await git_trainer.get_diff(repo_path, file_path)
    return {"suggestion": f"// Suggested change for {file_path}: {diff}"}
