import os
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
import litellm
from redis.asyncio import Redis
from mcp import tool
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

class LLMRequest(BaseModel):
    prompt: str
    model: str = "anthropic/claude-3.5-sonnet"
    max_tokens: int = 1000
    temperature: float = 0.7

@tool("llm_generate")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def generate_text(request: LLMRequest) -> Dict[str, Any]:
    """Generate text via LLM provider with Redis caching."""
    cache_key = f"llm:{request.model}:{request.prompt}"
    cached = await redis_client.get(cache_key)
    if cached:
        return {"text": cached.decode(), "source": "cache"}
    
    try:
        response = await litellm.acompletion(
            model=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            api_key=os.getenv(f"{request.model.split('/')[0].upper()}_API_KEY")
        )
        text = response.choices[0].message.content
        await redis_client.setex(cache_key, 3600, text)  # Cache for 1 hour
        return {"text": text, "source": "live"}
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        raise
