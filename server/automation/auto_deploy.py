from server.services.git_trainer import GitTrainer
from server.services.audit_log import AuditLog


class AutoDeploy:
    def __init__(self):
        self.git_trainer = GitTrainer()
        self.audit = AuditLog()

    async def deploy_to_github_pages(self, repo_name: str, branch: str = "main"):
        try:
            result = await self.git_trainer.execute_task(
                action="create_repo",
                params={"repo_name": repo_name, "private": False}
            )
            await self.audit.log_action(
                action="deploy_github_pages",
                user_id="system",
                details={"repo_name": repo_name, "branch": branch}
            )
            return {"status": "deployed", "repo": result["repo"]}
        except Exception as e:
            await self.audit.log_action(
                action="deploy_failed",
                user_id="system",
                details={"repo_name": repo_name, "error": str(e)}
            )
            return {"status": "failed", "error": str(e)}
