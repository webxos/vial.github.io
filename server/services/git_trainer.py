import httpx
from server.config import settings

class GitTrainer:
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers={"Authorization": f"Bearer {settings.GITHUB_TOKEN}"}
        )
        self.settings = settings

    async def execute_task(self, action: str, params: dict):
        if action == "create_repo":
            async with self.client as session:
                response = await session.post(
                    "/user/repos",
                    json={"name": params["repo_name"], "private": params.get("private", False)}
                )
                response.raise_for_status()
                return {"status": "created", "repo": response.json()}
        elif action == "commit_file":
            async with self.client as session:
                response = await session.put(
                    f"/repos/{self.settings.GITHUB_USERNAME}/{params['repo_name']}/contents/{params['file_path']}",
                    json={
                        "message": params["commit_message"],
                        "content": params["content"].encode("base64").decode("utf-8")
                    }
                )
                response.raise_for_status()
                return {"status": "committed"}
        else:
            raise ValueError("Invalid action")
