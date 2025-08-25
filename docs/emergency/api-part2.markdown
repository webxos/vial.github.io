# WebXOS 2025 Vial MCP SDK: API Emergency Backup - Part 2 (FastAPI REST Endpoints)

**Objective**: Implement advanced FastAPI REST endpoints for NASA, SpaceX, GitHub, and LangChain.

**Instructions for LLM**:
1. Update `server/main.py` to include all API routers.
2. Enhance `server/services/` with advanced endpoint logic.
3. Ensure OAuth2 with PKCE for secure access.
4. Integrate with existing services.

## Step 1: Update Main Application

### server/main.py
```python
from fastapi import FastAPI
from server.api.auth_endpoint import router as auth_router
from server.services.nasa_service import router as nasa_router
from server.services.spacex_service import router as spacex_router
from server.services.github_service import router as github_router
from server.services.langchain_service import router as langchain_router
from server.services.quantum_service import router as quantum_router
from server.services.rag_service import router as rag_router
from server.webxos_wallet import WebXOSWallet
from server.monitoring import setup_metrics
from server.models.base import init_db
from fastapi import Depends, HTTPException
from server.api.auth_endpoint import verify_token
from pydantic import BaseModel

app = FastAPI(title="WebXOS 2025 Vial MCP SDK", version="1.2.0")
app.include_router(auth_router)
app.include_router(nasa_router)
app.include_router(spacex_router)
app.include_router(github_router)
app.include_router(langchain_router)
app.include_router(quantum_router)
app.include_router(rag_router)
setup_metrics(app)
init_db()

# Wallet instance
wallet_manager = WebXOSWallet(password="secure_wallet_password")

class WalletCreate(BaseModel):
    address: str
    private_key: str
    balance: float = 0.0

@app.post("/mcp/wallet/create", tags=["wallet"])
async def create_wallet(wallet_data: WalletCreate, token: dict = Depends(verify_token)):
    try:
        wallet = wallet_manager.create_wallet(
            address=wallet_manager.sanitize_input(wallet_data.address),
            private_key=wallet_manager.sanitize_input(wallet_data.private_key),
            balance=wallet_data.balance
        )
        return {"address": wallet.address, "balance": wallet.balance}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/mcp/wallet/{address}", tags=["wallet"])
async def get_wallet(address: str, token: dict = Depends(verify_token)):
    address = wallet_manager.sanitize_input(address)
    if address not in wallet_manager.wallets:
        raise HTTPException(status_code=404, detail="Wallet not found")
    wallet = wallet_manager.wallets[address]
    return {"address": wallet.address, "balance": wallet.balance}
```

## Step 2: Enhance Service Files
Update `server/services/nasa_service.py` with advanced caching:
```python
import httpx
import os
from typing import Dict
from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token
from functools import lru_cache

class NASADataClient:
    def __init__(self):
        self.api_key = os.getenv("NASA_API_KEY")
        self.base_url = "https://api.nasa.gov"

    @lru_cache(maxsize=100)
    async def query_earthdata(self, bbox: tuple, temporal: str, collection: str) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/planetary/earth",
                params={"bbox": ",".join(map(str, bbox)), "temporal": temporal, "collection": collection, "api_key": self.api_key}
            )
            return response.json()

nasa_client = NASADataClient()

router = APIRouter(prefix="/mcp/nasa", tags=["nasa"])

@router.get("/earthdata")
async def get_earthdata(bbox: str, temporal: str, collection: str, token: dict = Depends(verify_token)):
    bbox_tuple = tuple(float(x) for x in bbox.split(","))
    return await nasa_client.query_earthdata(bbox_tuple, temporal, collection)
```

## Step 3: Validation
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
curl -H "Authorization: Bearer <token>" http://localhost:8000/mcp/nasa/earthdata?bbox=-180,-90,180,90&temporal=2023-01-01/2023-12-31&collection=MODIS
```

**Next**: Proceed to `api-part3.md` for quantum logic integration.