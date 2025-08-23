from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from httpx import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import logging
from fastapi.security import OAuth2AuthorizationCodeBearer

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

class LLMRequest(BaseModel):
    provider: str
    prompt: str

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def route_to_provider(provider: str, prompt: str) -> Dict:
    """Route request to specified LLM provider."""
    providers = {
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "mistral": os.getenv("MISTRAL_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "xai": os.getenv("XAI_API_KEY"),
        "meta": os.getenv("META_API_KEY"),
        "local": "http://localhost:8001"
    }
    if provider not in providers:
        raise HTTPException(status_code=400, detail="Invalid provider")
    
    async with AsyncClient() as client:
        # Placeholder: Actual API call to provider
        response = await client.post(
            providers[provider] if provider != "local" else providers["local"],
            json={"prompt": prompt}
        )
        response.raise_for_status()
        return response.json()

@router.post("/mcp/llm")
async def route_llm_request(request: LLMRequest, token: str = Security(oauth2_scheme)):
    """Route LLM request to provider."""
    try:
        result = await route_to_provider(request.provider, request.prompt)
        logger.info(f"LLM request routed to {request.provider}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"LLM routing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
