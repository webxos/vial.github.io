import json
from fastapi import FastAPI
from server.models.visual_components import ComponentModel
from server.services.advanced_logging import AdvancedLogger


class GitTrainer:
    def __init__(self):
        self.logger = AdvancedLogger()

    async def commit_visual_config(self, config: ComponentModel):
        generated_code = f"# Generated from visual config\nid: {config.id}\ntype: {config.type}"
        self.logger.log("Generated code from visual config", extra={"config_id": config.id})
        return {"status": "committed", "code": generated_code}

    async def deploy_to_github_pages(self, config: ComponentModel):
        self.logger.log("Deploying to GitHub Pages", extra={"config_id": config.id})
        return {"status": "deployed", "url": "https://yourusername.github.io/vial.github.io/"}

    async def fetch_training_data(self):
        data = {"training_data": "sample_data"}
        self.logger.log("Fetched training data", extra={"data_size": len(json.dumps(data))})
        return data


def setup_git_trainer(app: FastAPI):
    app.state.git_trainer = GitTrainer()
