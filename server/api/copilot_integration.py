from fastapi import APIRouter, Depends, HTTPException
from server.services.git_trainer import GitTrainer
from server.security import require_auth
from server.logging import logger
import requests

router = APIRouter()


@router.post("/copilot/generate")
async def generate_code(config: dict, user=Depends(require_auth)):
    try:
        git_trainer = GitTrainer()
        prompt = config.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt required")
        response = requests.post(
            "https://api.github.com/copilot/generate",
            headers={"Authorization": f"Bearer {settings.GITHUB_TOKEN}"},
            json={"prompt": prompt, "language": "python"}
        )
        response.raise_for_status()
        code = response.json().get("code")
        commit_result = await git_trainer.commit_with_mcp(
            message=f"Copilot: {prompt[:50]}",
            training_data={"code": code}
        )
        logger.log(f"Copilot code generated for user: {user['id']}")
        return {"code": code, "commit": commit_result}
    except Exception as e:
        logger.log(f"Copilot error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
