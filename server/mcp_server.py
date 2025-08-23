from fastapi import FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer
from server.api import health_check, visual_router, websocket, mcp_tools, webxos_wallet
from server.logging import logger
import os
import uvicorn
import uuid

app = FastAPI(title="Vial MCP Controller", version="2.9.3")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token")

app.include_router(health_check.router, prefix="/health")
app.include_router(visual_router.router, prefix="/visual")
app.include_router(websocket.router, prefix="/mcp")
app.include_router(mcp_tools.router, prefix="/mcp")
app.include_router(webxos_wallet.router, prefix="/wallet")

@app.on_event("startup")
async def startup_event():
    request_id = str(uuid.uuid4())
    try:
        logger.log("MCP server starting up", request_id=request_id)
        repo_path = os.getenv("REPO_PATH", ".")
        if not os.path.exists(repo_path):
            raise HTTPException(status_code=500, detail="Repository path not found")
        logger.log(f"Repository initialized at {repo_path}", request_id=request_id)
    except Exception as e:
        logger.log(f"Startup error: {str(e)}", request_id=request_id)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    request_id = str(uuid.uuid4())
    try:
        logger.log("MCP server shutting down", request_id=request_id)
    except Exception as e:
        logger.log(f"Shutdown error: {str(e)}", request_id=request_id)
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
