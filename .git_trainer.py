from octokit import Octokit
from server.config import get_settings
from server.logging import logger

class GitTrainer:
    def __init__(self):
        self.settings = get_settings()
        self.client = Octokit(auth='token', token=self.settings.GITHUB_TOKEN)

    def create_repo(self, repo_name: str, description: str = "", private: bool = False):
        try:
            response = self.client.repos.create_for_authenticated_user(
                name=repo_name,
                description=description,
                private=private
            )
            logger.info(f"Created repository: {repo_name}")
            return {"status": "created", "repo": response.json}
        except Exception as e:
            logger.error(f"Failed to create repository: {str(e)}")
            raise ValueError(f"Repository creation failed: {str(e)}")

    def commit_file(self, repo_name: str, file_path: str, content: str, commit_message: str):
        try:
            response = self.client.repos.create_or_update_file_contents(
                owner=self.settings.GITHUB_USERNAME,
                repo=repo_name,
                path=file_path,
                message=commit_message,
                content=content.encode('base64').decode('utf-8')
            )
            logger.info(f"Committed file {file_path} to {repo_name}")
            return {"status": "committed", "commit": response.json}
        except Exception as e:
            logger.error(f"Failed to commit file: {str(e)}")
            raise ValueError(f"Commit failed: {str(e)}")

git_trainer = GitTrainer()
