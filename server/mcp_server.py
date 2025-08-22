# server/mcp_server.py
from fastapi import FastAPI
from server.api import webxos_wallet, quantum_endpoints, comms_hub
from server.config.final_config import settings
from server.services.advanced_logging import AdvancedLogging

app = FastAPI()

# Include API routers
app.include_router(webxos_wallet.router, prefix="/wallet")
app.include_router(quantum_endpoints.router, prefix="/quantum")
app.include_router(comms_hub.router, prefix="/comms")

# Setup logging middleware
AdvancedLogging(app)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "database": settings.SQLALCHEMY_DATABASE_URL,
            "reputation_logging": settings.REPUTATION_LOGGING_ENABLED
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
