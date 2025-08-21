from fastapi import APIRouter, Depends
from server.services.git_trainer import GitTrainer
from server.security import verify_jwt


router = APIRouter()


@router.post("/copilot/suggest")
async def suggest_code(
    code_snippet: dict,
    token: str = Depends(verify_jwt)
):
    git_trainer = GitTrainer()
    try:
        suggestion = await git_trainer.execute_task(
            action="suggest_code",
            params={"code_snippet": code_snippet}
        )
        return {"status": "success", "suggestion": suggestion}
    except Exception as e:
        return {"status": "error", "message": str(e)}
