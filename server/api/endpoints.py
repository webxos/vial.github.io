from fastapi import APIRouter, Depends
from server.services.git_trainer import git_trainer
from server.security import verify_token

router = APIRouter()


async def create_repository(repo_data: dict, token: str = Depends(verify_token)):
    result = git_trainer.create_repo(
        repo_data["name"],
        repo_data.get("description", ""),
        repo_data.get("private", False)
    )
    return result


async def commit_file(commit_data: dict, token: str = Depends(verify_token)):
    result = git_trainer.commit_file(
        commit_data["repo_name"],
        commit_data["file_path"],
        commit_data["content"],
        commit_data["commit_message"]
    )
    return result
