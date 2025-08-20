from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from jose import jwt
from server.security import verify_token, generate_credentials, get_password_hash, verify_password, SECRET_KEY, ALGORITHM
import os
from datetime import datetime, timedelta

app = FastAPI(title="Vial MCP Controller")

# Mount static files
app.mount("/public", StaticFiles(directory="public"), name="public")

# JSON-RPC Request Model
class JsonRpcRequest(BaseModel):
    jsonrpc: str
    method: str
    params: dict = {}
    id: int

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# JSON-RPC endpoint
@app.post("/jsonrpc")
async def jsonrpc_endpoint(request: JsonRpcRequest, token: str = Depends(verify_token)):
    if request.jsonrpc != "2.0":
        raise HTTPException(status_code=400, detail="Invalid JSON-RPC version")
    
    # Handle methods
    if request.method == "status":
        return {"jsonrpc": "2.0", "result": {"status": "running", "version": "2.7"}, "id": request.id}
    elif request.method == "help":
        return {
            "jsonrpc": "2.0",
            "result": {
                "commands": ["/login", "/train", "/translate", "/diagnose", "/sync", "/status", "/wallet", "/deploy"]
            },
            "id": request.id
        }
    else:
        raise HTTPException(status_code=400, detail="Method not found")

# OAuth token endpoint
@app.post("/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Placeholder: Replace with actual user validation
    if form_data.username == "admin" and verify_password(form_data.password, get_password_hash("admin")):
        token = jwt.encode(
            {"sub": form_data.username, "exp": datetime.utcnow() + timedelta(hours=1)},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# Generate API credentials
@app.post("/auth/generate-credentials")
async def generate_api_credentials(token: str = Depends(verify_token)):
    key, secret = generate_credentials()
    return {"key": key, "secret": secret}
