from git import Repo
from server.models.mcp_alchemist import mcp_alchemist
from server.config import get_settings
import os

class GitTrainer:
    def __init__(self):
        self.settings = get_settings()
        self.repo_dir = "./training_repo"

    def clone_repo(self, repo_url: str):
        if os.path.exists(self.repo_dir):
            repo = Repo(self.repo_dir)
        else:
            repo = Repo.clone_from(repo_url, self.repo_dir)
        return repo

    def train_from_repo(self, repo_url: str, data_path: str):
        repo = self.clone_repo(repo_url)
        data_file = os.path.join(self.repo_dir, data_path)
        if not os.path.exists(data_file):
            raise ValueError("Data file not found in repository")
        
        with open(data_file, "r") as f:
            data = f.read()
        
        # Train alchemist with data
        result = mcp_alchemist.train(data)
        repo.index.add([data_path])
        repo.index.commit("Trained model data")
        repo.remotes.origin.push()
        
        return {"status": "training complete", "commit": repo.head.commit.hexsha}

git_trainer = GitTrainer()
