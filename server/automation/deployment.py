from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.services.git_trainer import GitTrainer
from server.services.advanced_logging import AdvancedLogger
from server.models.visual_components import VisualConfig
from pydantic import BaseModel


class DeployRequest(BaseModel):
    config_id: str


def setup_deployment(app: FastAPI):
    logger = AdvancedLogger()

    @app.post("/deploy-config")
    async def deploy_config(request: DeployRequest, db: Session = Depends(get_db), git_trainer: GitTrainer = Depends(lambda: GitTrainer())):
        config = db.query(VisualConfig).filter(VisualConfig.id == request.config_id).first()
        if not config:
            logger.log("Deployment failed: Config not found", extra={"config_id": request.config_id})
            return {"error": "Config not found"}
        for component in config.components:
            await git_trainer.commit_visual_config(ComponentModel(**component))
        result = await git_trainer.deploy_to_github_pages(ComponentModel(**config.components[0]))
        logger.log("Configuration deployed", extra={"config_id": request.config_id})
        return {"status": "deployed", "url": result["url"]}
