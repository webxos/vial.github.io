from fastapi import APIRouter, Depends
from server.models.mcp_alchemist import mcp_alchemist
from server.security import verify_token

router = APIRouter()


async def process_alchemist_request(request: dict, token: str = Depends(verify_token)):
    result = mcp_alchemist.process(request)
    return result


async def train_alchemist_model(data: dict, token: str = Depends(verify_token)):
    result = mcp_alchemist.train(data)
    return result
