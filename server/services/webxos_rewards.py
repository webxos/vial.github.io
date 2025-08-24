from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from server.utils.security_sanitizer import sanitize_input
from fastapi.security import OAuth2AuthorizationCodeBearer
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

class RewardsRequest(BaseModel):
    user_id: str
    action: str
    metadata: dict

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def calculate_rewards(user_id: str, action: str, metadata: dict) -> dict:
    """Calculate WebXOS rewards for user actions."""
    try:
        sanitized_user_id = sanitize_input(user_id)
        sanitized_action = sanitize_input(action)
        sanitized_metadata = sanitize_input(metadata)
        
        # Placeholder: Blockchain-based rewards calculation
        rewards = {"points": 100, "action": sanitized_action}
        
        logger.info(f"Rewards calculated for user {sanitized_user_id}: {sanitized_action}")
        return {"status": "success", "rewards": rewards}
    except Exception as e:
        logger.error(f"Rewards calculation failed: {str(e)}")
        raise

@router.post("/mcp/rewards")
async def process_rewards(request: RewardsRequest, token: str = Security(oauth2_scheme)):
    """Process WebXOS rewards for user actions."""
    try:
        result = await calculate_rewards(request.user_id, request.action, request.metadata)
        return result
    except Exception as e:
        logger.error(f"Rewards processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
