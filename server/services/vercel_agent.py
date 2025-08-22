from fastapi import HTTPException
from server.config import settings
from server.logging import logger
import requests


class VercelAgent:
    def __init__(self):
        self.api_token = settings.VERCEL_API_TOKEN
        self.project_id = settings.VERCEL_PROJECT_ID
        self.base_url = "https://api.vercel.com/v9"

    async def deploy_project(self, config: dict):
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            payload = {
                "name": "vial-mcp",
                "files": config.get("files", []),
                "projectId": self.project_id,
                "target": "production"
            }
            response = requests.post(
                f"{self.base_url}/projects/{self.project_id}/deployments",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            logger.log(f"Vercel deployment initiated: {response.json()['id']}")
            return {"status": "deployed", "deployment_id": response.json()["id"]}
        except Exception as e:
            logger.log(f"Vercel deployment error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_deployment_status(self, deployment_id: str):
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            response = requests.get(
                f"{self.base_url}/deployments/{deployment_id}",
                headers=headers
            )
            response.raise_for_status()
            status = response.json().get("status")
            logger.log(f"Vercel deployment status: {status}")
            return {"status": status}
        except Exception as e:
            logger.log(f"Vercel status check error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
