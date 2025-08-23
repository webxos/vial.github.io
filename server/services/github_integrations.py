from typing import Dict, Any
import requests
from server.logging_config import logger
import os
import uuid

class GitHubIntegration:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github.v3+json"}
        self.repo = os.getenv("GITHUB_REPO", "vial/vial-mcp")

    async def fork_repository(self, request_id: str) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"https://api.github.com/repos/{self.repo}/forks",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"Forked repository {self.repo}", request_id=request_id)
            return {"fork_url": data["html_url"], "request_id": request_id}
        except Exception as e:
            logger.error(f"Fork error: {str(e)}", request_id=request_id)
            raise

    async def commit_training_results(self, vial_id: str, request_id: str) -> Dict[str, Any]:
        try:
            content = f"Training results for {vial_id}\nTimestamp: 2025-08-23T02:20:00Z"
            response = requests.put(
                f"https://api.github.com/repos/{self.repo}/contents/training/{vial_id}.md",
                headers=self.headers,
                json={
                    "message": f"Commit training results for {vial_id}",
                    "content": content.encode("base64").decode("utf-8"),
                    "branch": "main"
                }
            )
            response.raise_for_status()
            logger.info(f"Committed training results for {vial_id}", request_id=request_id)
            return {"commit_url": response.json()["content"]["html_url"], "request_id": request_id}
        except Exception as e:
            logger.error(f"Commit error: {str(e)}", request_id=request_id)
            raise
