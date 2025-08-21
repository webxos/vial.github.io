from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer
from server.services.git_trainer import git_trainer

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


@router.post("/repos/create")
async def create_repo(repo_name: str, description: str = "", private: bool = False,
                      token: str = Depends(oauth2_scheme)):
    return git_trainer.create_repo(repo_name, description, private)


@router.post("/repos/commit")
async def commit_file(repo_name: str, file_path: str, content: str,
                      commit_message: str, token: str = Depends(oauth2_scheme)):
    return git_trainer.commit_file(
        repo_name, file_path, content, commit_message
    )
