from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from server.api import auth, endpoints, health_check, quantum_endpoints
from server.services.notification import notification_service
from server.logging import logger
import json

app = FastAPI(
    title="Vial MCP Controller",
    description="AI-driven task management with GitHub, MongoDB, Redis, and Qiskit",
    version="1.0.0"
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


@app.get("/health")
async def health():
    return await health_check.check_health()


@app.post("/jsonrpc")
async def jsonrpc(request: dict, token: str = Depends(oauth2_scheme)):
    if request.get("jsonrpc") != "2.0":
        raise HTTPException(status_code=400, detail="Invalid JSON-RPC version")
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    try:
        if method == "status":
            return {"jsonrpc": "2.0", "result": {"status": "ok"}, "id": request_id}
        elif method == "help":
            return {
                "jsonrpc": "2.0",
                "result": {
                    "methods": [
                        {"name": "status", "description": "Check server status"},
                        {"name": "help", "description": "List available methods"}
                    ]
                },
                "id": request_id
            }
        else:
            raise HTTPException(status_code=400, detail="Method not found")
    except Exception as e:
        logger.error(f"JSON-RPC error: {str(e)}")
        return {"jsonrpc": "2.0", "error": {"code": -32600, "message": str(e)}, "id": request_id}


app.include_router(auth.router, prefix="/auth")
app.include_router(endpoints.router)
app.include_router(quantum_endpoints.router, prefix="/quantum")


@app.on_event("startup")
async def startup_event():
    await notification_service.send_notification(
        "admin", "Server started", "inapp"
    )


@app.on_event("shutdown")
async def shutdown_event():
    await notification_service.send_notification(
        "admin", "Server stopped", "inapp"
    )
