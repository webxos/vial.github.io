from fastapi import FastAPI
from git import Repo
from server.mcp_server import app


class GitTrainer:
    def __init__(self):
        self.repo = Repo(".")

    def commit_changes(self, message: str):
        self.repo.git.add(all=True)
        self.repo.index.commit(message)

    async def get_diff(self, repo_path: str, file_path: str):
        repo = Repo(repo_path)
        return repo.git.diff(file_path)


def setup_git_trainer(app: FastAPI):
    trainer = GitTrainer()
    app.state.git_trainer = trainer
