from fastapi import APIRouter
from ..extensions.prompt_manager import prompt_manager

router = APIRouter()

@router.post("/execute-prompt")
async def execute_prompt(command: str):
    result = prompt_manager.execute_prompt(command)
    return {"result": result}
