from server.sdk.vial_sdk import vial_sdk
from server.config import get_settings
import docker
import subprocess
import os

class AutoDeploy:
    def __init__(self):
        self.settings = get_settings()
        self.docker_client = docker.from_env()

    def deploy(self, service: str = "app"):
        # Initialize system with SDK
        vial_sdk.initialize_system({"username": "system"})

        # Run docker-compose
        try:
            subprocess.run(
                ["docker-compose", "-f", "docker-compose.yml", "up", "-d", service],
                check=True,
                cwd=os.path.dirname(os.path.abspath(__file__ + "/../../"))
            )
            return {"status": f"{service} deployed"}
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Deployment failed: {str(e)}")

    def stop(self, service: str = "app"):
        try:
            subprocess.run(
                ["docker-compose", "-f", "docker-compose.yml", "stop", service],
                check=True,
                cwd=os.path.dirname(os.path.abspath(__file__ + "/../../"))
            )
            return {"status": f"{service} stopped"}
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Stop failed: {str(e)}")

auto_deploy = AutoDeploy()
