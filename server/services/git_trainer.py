from fastapi import FastAPI
from git import Repo
from server.config.settings import settings
from server.logging import logger

class GitTrainer:
    def __init__(self):
        self.repo = Repo(".")

    def commit_changes(self, message: str):
        self.repo.git.add(all=True)
        self.repo.index.commit(message)
        logger.log(f"Committed changes: {message}")

    def save_training_data(self, data: dict):
        with open("training_data.json", "w") as f:
            json.dump(data, f)
        self.repo.git.add("training_data.json")
        self.repo.index.commit(f"Training data saved: {data}")

def setup_git_trainer(app: FastAPI):
    trainer = GitTrainer()
    app.state.git_trainer = trainer

    @app.post("/git/commit")
    async def commit_endpoint(message: str):
        app.state.git_trainer.commit_changes(message)
        return {"status": "committed"}
