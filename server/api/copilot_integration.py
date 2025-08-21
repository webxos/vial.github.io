from fastapi import APIRouter, Depends
from server.config import copilot_config
from server.security import verify_token

router = APIRouter()


async def get_copilot_suggestions(query: dict, token: str = Depends(verify_token)):
    config = copilot_config.get_config()
    suggestions = copilot_config.generate_suggestions(query, config)
    return {"suggestions": suggestions}
