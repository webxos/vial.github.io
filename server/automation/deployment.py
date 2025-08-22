from fastapi import FastAPI
import subprocess
from server.config import config

class DeploymentAutomator:
    def __init__(self, app: FastAPI):
        self.app = app

    def deploy_to_vps(self):
        if config.VPS_IP:
            try:
                subprocess.run(["ssh", f"user@{config.VPS_IP}", "docker-compose up -d"], check=True)
                return {"status": "deployed"}
            except subprocess.CalledProcessError:
                return {"status": "deployment_failed"}
        return {"status": "no_vps_config"}

def setup_deployment(app: FastAPI):
    deployer = DeploymentAutomator(app)
    app.state.deployer = deployer

    @app.post("/automation/deploy")
    async def deploy_endpoint():
        return deployer.deploy_to_vps()
