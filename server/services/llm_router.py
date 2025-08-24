```python
from typing import Dict, List, Optional
from fastapi import HTTPException
from pydantic import BaseModel
import logging
from litellm import completion
from server.config.settings import settings
from server.utils.security_sanitizer import sanitize_input, validate_prompt

logger = logging.getLogger(__name__)

class LLMRequest(BaseModel):
    provider: str
    prompt: str
    tools: Optional[List[str]] = None
    model: Optional[str] = None

class LLMRouter:
    def __init__(self):
        self.providers = {
            "anthropic": {"api_key": settings.ANTHROPIC_API_KEY, "default_model": "claude-3.5-sonnet"},
            "mistral": {"api_key": settings.MISTRAL_API_KEY, "default_model": "mistral-large"},
            "google": {"api_key": settings.GOOGLE_API_KEY, "default_model": "gemini-1.5-flash"},
            "xai": {"api_key": settings.OPENAI_API_KEY, "base_url": settings.OPENAI_BASE_URL, "default_model": "grok-3"},
            "meta": {"api_key": settings.OPENAI_API_KEY, "base_url": settings.OPENAI_BASE_URL, "default_model": "llama-3"},
            "local": {"base_url": settings.LOCAL_LLM_URL, "default_model": "llama-3"}
        }

    async def route_request(self, request: LLMRequest) -> Dict:
        """Route LLM request to the specified provider."""
        try:
            sanitized_prompt = sanitize_input(validate_prompt(request.prompt))
            provider_config = self.providers.get(request.provider)
            if not provider_config:
                raise HTTPException(status_code=400, detail=f"Invalid provider: {request.provider}")

            model = request.model or provider_config["default_model"]
            params = {
                "model": f"{request.provider}/{model}",
                "messages": [{"role": "user", "content": sanitized_prompt}],
                "tools": request.tools or [],
                "max_tokens": 4096
            }

            if "api_key" in provider_config:
                params["api_key"] = provider_config["api_key"]
            if "base_url" in provider_config:
                params["api_base"] = provider_config["base_url"]

            response = await completion(**params)
            logger.info(f"LLM request routed to {request.provider}/{model}")
            return response.to_dict()
        except Exception as e:
            logger.error(f"LLM routing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
```
