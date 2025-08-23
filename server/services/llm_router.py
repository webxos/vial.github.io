import logging
from typing import Dict, Any
from pydantic import BaseModel
from httpx import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential
from server.auth.rbac import check_rbac

logger = logging.getLogger(__name__)

class LLMRequest(BaseModel):
    provider: str
    prompt: str

class LLMRouter:
    def __init__(self):
        self.providers = {
            "anthropic": "https://api.anthropic.com/v1",
            "mistral": "https://api.mixtral.ai/v1",
            "google": "https://generativelanguage.googleapis.com/v1",
            "xai": "https://api.x.ai/v1",
            "meta": "https://api.meta.ai/v1",
            "local": "http://localhost:8001"
        }
        # Placeholder: OBS/SVG video integration for real-time LLM visualization
        # Future: Add /obs/stream for LLM-driven SVG rendering

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def route_request(self, provider: str, prompt: str) -> Dict[str, Any]:
        """Route LLM request to specified provider with retry logic."""
        try:
            async with AsyncClient() as client:
                shield_response = await client.post(
                    "https://api.azure.ai/content-safety/prompt-shields",
                    json={"prompt": prompt}
                )
                if shield_response.json().get("malicious"):
                    raise ValueError("Malicious input detected")

                if provider not in self.providers:
                    raise ValueError(f"Unsupported provider: {provider}")

                response = await client.post(
                    f"{self.providers[provider]}/generate",
                    json={"prompt": prompt, "max_tokens": 512},
                    headers={"Authorization": f"Bearer {os.getenv(f'{provider.upper()}_API_KEY', 'mock_key')}"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"LLM routing failed for {provider}: {str(e)}")
            raise
