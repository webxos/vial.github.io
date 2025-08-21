import httpx
from server.config import settings
from server.services.audit_log import AuditLog


class GitTrainer:
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers={"Authorization": f"Bearer {settings.GITHUB_TOKEN}"}
        )
        self.settings = settings
        self.audit = AuditLog()

    async def execute_task(self, action: str, params: dict):
        if action == "create_repo":
            async with self.client as session:
                response = await session.post(
                    "/user/repos",
                    json={
                        "name": params["repo_name"],
                        "private": params.get("private", False)
                    }
                )
                response.raise_for_status()
                await self.audit.log_action(
                    action="create_repo",
                    user_id=self.settings.GITHUB_USERNAME,
                    details={"repo_name": params["repo_name"]}
                )
                return {"status": "created", "repo": response.json()}
        elif action == "commit_file":
            async with self.client as session:
                response = await session.put(
                    f"/repos/{self.settings.GITHUB_USERNAME}/"
                    f"{params['repo_name']}/contents/{params['file_path']}",
                    json={
                        "message": params["commit_message"],
                        "content": params["content"].encode("base64").decode("utf-8")
                    }
                )
                response.raise_for_status()
                await self.audit.log_action(
                    action="commit_file",
                    user_id=self.settings.GITHUB_USERNAME,
                    details={"repo_name": params["repo_name"], "file_path": params["file_path"]}
                )
                return {"status": "committed"}
        elif action == "suggest_code":
            # Placeholder for Copilot-like suggestion logic
            suggestion = {"code": params["code_snippet"]["code"] + " # Suggested by Copilot"}
            await self.audit.log_action(
                action="suggest_code",
                user_id=self.settings.GITHUB_USERNAME,
                details={"code_snippet": params["code_snippet"]}
            )
            return suggestion
        else:
            raise ValueError("Invalid action")
