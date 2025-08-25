import httpx
import os
from typing import Dict
from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token

class GitHubService:
    def __init__(self):
        self.base_url = os.getenv("GITHUB_HOST", "https://api.githubcopilot.com")
        self.token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

    async def get_repo(self, owner: str, repo: str) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            response.raise_for_status()
            return response.json()

github_service = GitHubService()

router = APIRouter(prefix="/mcp/github", tags=["github"])

@router.get("/repos/{owner}/{repo}")
async def get_repo(owner: str, repo: str, token: dict = Depends(verify_token)) -> Dict:
    return await github_service.get_repo(owner, repo)
