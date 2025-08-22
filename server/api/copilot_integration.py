from fastapi import APIRouter, Depends
from server.services.git_trainer import GitTrainer
from server.services.advanced_logging import AdvancedLogger
from server.models.visual_components import ComponentModel
from pydantic import BaseModel


class CopilotRequest(BaseModel):
    component: ComponentModel


router = APIRouter()
logger = AdvancedLogger()


@router.post("/generate-code")
async def generate_code(request: CopilotRequest, git_trainer: GitTrainer = Depends(lambda: GitTrainer())):
    try:
        result = await git_trainer.commit_visual_config(request.component)
        logger.log("Code generated via Copilot", extra={"component_id": request.component.id})
        return {"status": "generated", "code": result["code"]}
    except Exception as e:
        logger.log("Code generation failed", extra={"error": str(e)})
        return {"error": str(e)}
