from server.services.git_trainer import git_trainer
from server.logging import logger
import docker


class AutoDeploy:
    def __init__(self):
        self.docker_client = docker.from_env()

    def deploy(self, repo_name: str):
        try:
            git_trainer.create_repo(repo_name)
            self.docker_client.containers.run(
                "vial-mcp-alchemist:latest",
                detach=True
            )
            logger.info(f"Deployed {repo_name}")
            return {"status": "deployed"}
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise ValueError(f"Deployment failed: {str(e)}")


auto_deploy = AutoDeploy()
