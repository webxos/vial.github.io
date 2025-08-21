from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from server.security import verify_token, generate_credentials
from server.api.auth import router as auth_router
from server.api.endpoints import router as endpoints_router
from server.api.quantum_endpoints import router as quantum_router
from server.api.alchemist_endpoints import router as alchemist_router
from server.api.copilot_integration import router as copilot_router
from server.api.health_check import router as health_router
from server.api.websocket import router as websocket_router
from server.api.docs import router as docs_router
from server.error_handler import exception_handler
from server.api.rate_limiter import rate_limit_middleware
from server.api.middleware import cors_middleware
from server.api.cache_control import cache_middleware

app = FastAPI(title="Vial MCP Controller")


app.exception_handler(Exception)(exception_handler)


app.middleware("http")(rate_limit_middleware)
app.middleware("http")(cors_middleware)
app.middleware("http")(cache_middleware)


app.mount("/public", StaticFiles(directory="public"), name="public")


app.include_router(auth_router)
app.include_router(endpoints_router)
app.include_router(quantum_router)
app.include_router(alchemist_router)
app.include_router(copilot_router)
app.include_router(health_router)
app.include_router(websocket_router)
app.include_router(docs_router)


class JsonRpcRequest(BaseModel):
    jsonrpc: str
    method: str
    params: dict = {}
    id: int


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/jsonrpc")
async def jsonrpc_endpoint(request: JsonRpcRequest, token: str = Depends(verify_token)):
    if request.jsonrpc != "2.0":
        raise HTTPException(status_code=400, detail="Invalid JSON-RPC version")
    if request.method == "status":
        return {"jsonrpc": "2.0", "result": {"status": "running", "version": "2.7"},
                "id": request.id}
    elif request.method == "help":
        return {
            "jsonrpc": "2.0",
            "result": {
                "commands": ["/login", "/train", "/translate", "/diagnose", "/sync",
                             "/wallet", "/deploy", "/copilot", "/health_check"]
            },
            "id": request.id
        }
    else:
        raise HTTPException(status_code=400, detail="Method not found")


@app.post("/auth/generate-credentials")
async def generate_api_credentials(token: str = Depends(verify_token)):
    key, secret = generate_credentials()
    return {"key": key, "secret": secret}
