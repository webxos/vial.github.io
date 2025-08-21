from fastapi import FastAPI
from server.api import auth, endpoints, quantum_endpoints, websocket, copilot_integration, jsonrpc, void, troubleshoot, help
from server.api.cache_control import cache_response
from server.api.rate_limiter import rate_limit
from server.api.middleware import logging_middleware
from server.api.error_handler import setup_error_handlers
from server.services.security import setup_cors


app = FastAPI(
    title="Vial MCP Controller",
    description="Modular control plane for AI-driven task management",
    version="2.7.0",
)


app.middleware("http")(cache_response)
app.middleware("http")(rate_limit)
app.middleware("http")(logging_middleware)
setup_cors(app)
setup_error_handlers(app)


app.include_router(auth.router, prefix="/auth")
app.include_router(endpoints.router)
app.include_router(quantum_endpoints.router, prefix="/quantum")
app.include_router(websocket.router)
app.include_router(copilot_integration.router, prefix="/copilot")
app.include_router(jsonrpc.router, prefix="/jsonrpc")
app.include_router(void.router)
app.include_router(troubleshoot.router)
app.include_router(help.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
