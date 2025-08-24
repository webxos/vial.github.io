from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2AuthorizationCodeBearer
from server.api import auth_endpoint, quantum_rag_endpoint, servicenow_endpoint, monitoring_endpoint, alchemist_endpoint
from server.services import obs_handler
from server.config.settings import Settings
from server.config.database import get_db
from sqlalchemy.orm import Session
import logging

app = FastAPI(title="Vial MCP Controller", version="1.0.0")
settings = Settings()
logging.basicConfig(level=logging.INFO, filename="logs/server.log")

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="/mcp/auth/authorize",
    tokenUrl="/mcp/auth/token",
    scheme_name="OAuth2PKCE"
)

# Global error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logging.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unexpected error: {str(exc)}")
    return {"error": "Internal server error", "status_code": 500}

# Include API routers
app.include_router(auth_endpoint.router, prefix="/mcp/auth")
app.include_router(quantum_rag_endpoint.router, prefix="/mcp/quantum_rag")
app.include_router(servicenow_endpoint.router, prefix="/mcp/servicenow")
app.include_router(monitoring_endpoint.router, prefix="/mcp/monitoring")
app.include_router(alchemist_endpoint.router, prefix="/mcp/alchemist")

# OBS endpoint
@app.post("/mcp/tools/obs.init")
async def init_obs_scene(scene_name: str, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        result = await obs_handler.init_obs_scene(scene_name, settings.obs_host, settings.obs_port, settings.obs_password)
        return {"scene": result}
    except Exception as e:
        logging.error(f"OBS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
