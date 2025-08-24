```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from server.api.quantum_rag_endpoint import router as quantum_router
from server.api.servicenow_endpoint import router as servicenow_router
from server.api.monitoring_endpoint import router as monitoring_router
from server.api.auth_endpoint import router as auth_router
from server.services.obs_handler import OBSHandler

app = FastAPI(title="Vial MCP Controller")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://vial.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Include routers
app.include_router(quantum_router)
app.include_router(servicenow_router)
app.include_router(monitoring_router)
app.include_router(auth_router)

# OBS endpoint
@app.post("/mcp/tools/obs.init")
async def init_obs_scene(request: dict):
    """Initialize OBS scene."""
    try:
        scene_name = request.get("scene_name")
        if not scene_name:
            raise HTTPException(status_code=400, detail="scene_name is required")
        obs = OBSHandler()
        result = await obs.init_scene(scene_name)
        obs.disconnect()
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"OBS init failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Global exception handler for traffic management
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Request failed: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )

@app.get("/")
async def root():
    return {"message": "Vial MCP Controller API"}
```
