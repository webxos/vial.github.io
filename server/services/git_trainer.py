from git import Repo
from server.services.vial_manager import VialManager
from server.models.visual_components import VisualConfig
from server.logging import logger
import os


class GitTrainer:
    def __init__(self):
        self.repo = Repo(os.getcwd())
        self.vial_manager = VialManager()

    async def commit_visual_config(self, config: VisualConfig):
        try:
            # Generate FastAPI code from visual config
            code = self._generate_fastapi_code(config)
            output_path = "generated_routes.py"
            with open(output_path, "w") as f:
                f.write(code)
            self.repo.git.add(output_path)
            commit_message = f"MCP: Visual config commit {config.components[0].id}"
            self.repo.index.commit(commit_message)
            self.repo.remotes.origin.push()
            logger.log(f"Committed visual config: {commit_message}")
            return {"status": "committed", "commit_message": commit_message}
        except Exception as e:
            logger.log(f"Git commit error: {str(e)}")
            return {"error": str(e)}

    def _generate_fastapi_code(self, config: VisualConfig) -> str:
        code = ["from fastapi import APIRouter\n", "router = APIRouter()\n"]
        for component in config.components:
            if component.type == "api_endpoint":
                route = f"""
@router.{component.config.get('method', 'get').lower()}("/{component.id}")
async def {component.id.replace('-', '_')}():
    return {{"status": "executed", "endpoint": "{component.id}"}}"""
                code.append(route)
        return "\n".join(code)
