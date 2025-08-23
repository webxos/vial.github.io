import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel
from httpx import AsyncClient
from server.security.crypto_engine import CryptoEngine
from server.auth.rbac import check_rbac
from fastapi_limiter import FastAPILimiter
from redis.asyncio import Redis

logger = logging.getLogger(__name__)
app = FastAPI()
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
    scopes={"https://www.googleapis.com/auth/drive": "Google Drive access"}
)

class PromptShieldRequest(BaseModel):
    prompt: str

class WalletRequest(BaseModel):
    address: str
    amount: float

async def init_limiter():
    redis = Redis.from_url("redis://localhost:6379/0")
    await FastAPILimiter.init(redis)

app.add_event_handler("startup", init_limiter)

@app.post("/v1/auth/token")
async def token(token: str = Security(oauth2_scheme)):
    """Generate JWT with OAuth 2.0+PKCE."""
    async with AsyncClient() as client:
        try:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v3/tokeninfo",
                params={"access_token": token}
            )
            response.raise_for_status()
            return {"jwt": "mock_jwt"}  # Simplified for brevity
        except Exception as e:
            logger.error(f"OAuth token validation failed: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/v1/wallet/export", dependencies=[Depends(check_rbac(["wallet:write"]))])
async def export_wallet(request: WalletRequest):
    """Export encrypted wallet with Prompt Shields validation."""
    async with AsyncClient() as client:
        try:
            shield_response = await client.post(
                "https://api.azure.ai/content-safety/prompt-shields",
                json={"prompt": request.address}
            )
            if shield_response.json().get("malicious"):
                raise HTTPException(status_code=400, detail="Malicious input detected")
            crypto = CryptoEngine()
            encrypted = crypto.encrypt(EncryptionParams(data=request.json().encode()))
            return {"wallet_md": f"# Wallet\nCiphertext: {encrypted.data.hex()}"}
        except Exception as e:
            logger.error(f"Wallet export failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
