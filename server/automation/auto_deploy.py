import os
import subprocess
from fastapi import HTTPException
from server.config import settings
from server.logging import logger


class AutoDeploy:
    def __init__(self):
        self.deploy_env = os.getenv("DEPLOY_ENV", "local")

    async def deploy(self, platform: str, config: dict):
        try:
            if platform == "netlify":
                cmd = [
                    "netlify", "deploy", "--prod",
                    f"--dir={config.get('dir', 'dist')}"
                ]
            elif platform == "vercel":
                cmd = [
                    "vercel", "--prod",
                    f"--token={settings.VERCEL_TOKEN}"
                ]
            else:
                raise HTTPException(status_code=400, detail="Invalid platform")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            logger.info(f"Deployed to {platform}: {result.stdout}")
            return {"status": "deployed", "platform": platform}
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {e.stderr}")
            raise HTTPException(status_code=500, detail="Deployment failed")


auto_deploy = AutoDeploy()
