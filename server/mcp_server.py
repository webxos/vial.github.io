from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from server.api import (
    auth, quantum_endpoints, websocket, stream, webxos_wallet,
    comms_hub, yaml_workflow
)
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid
import os
from fastapi.security import OAuth2PasswordBearer

app = FastAPI(
    title="Vial MCP Controller",
    description="API for managing vials, wallets, and quantum tasks",
    version="2.7"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@app.get("/health")
async def health_check(token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        alchemist = Alchemist()
        db_status = await alchemist.check_db_connection()
        agent_status = await alchemist.check_agent_availability()
        wallet_status = await alchemist.check_wallet_system()
        response_time = await alchemist.get_api_response_time()
        logger.info(
            f"Health check: DB={db_status}, Agents={agent_status}, Wallet={wallet_status}",
            request_id=request_id
        )
        return {
            "status": "healthy",
            "db": db_status,
            "agents": agent_status,
            "wallet": wallet_status,
            "response_time": response_time,
            "request_id": request_id
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", request_id=request_id)
        return {"status": "unhealthy", "detail": str(e), "request_id": request_id}

app.include_router(auth.router, prefix="/auth")
app.include_router(quantum_endpoints.router, prefix="/quantum")
app.include_router(websocket.router, prefix="/mcp")
app.include_router(stream.router, prefix="/stream")
app.include_router(webxos_wallet.router, prefix="/wallet")
app.include_router(comms_hub.router, prefix="/comms")
app.include_router(yaml_workflow.router, prefix="/yaml")
