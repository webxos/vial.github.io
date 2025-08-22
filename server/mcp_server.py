from fastapi import FastAPI
from server.api import auth, endpoints, quantum_endpoints, websocket, copilot_integration, jsonrpc, void, troubleshoot, help, comms_hub, upload, stream
from server.api.cache_control import cache_response
from server.api.rate_limiter import rate_limit
from server.api.middleware import logging_middleware
from server.api.error_handler import setup_error_handlers
from server.services.security import setup_cors
from server.api.security_headers import setup_security_headers
from server.services.agent_tasks import setup_agent_tasks
from server.services.prompt_training import setup_prompt_training
from server.services.training_scheduler import setup_training_scheduler
from server.services.advanced_logging import setup_advanced_logging
from server.services.error_recovery import setup_error_recovery


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
setup_security_headers(app)
setup_agent_tasks(app)
setup_prompt_training(app)
setup_training_scheduler(app)
setup_advanced_logging(app)
setup_error_recovery(app)

app.include_router(auth.router, prefix="/auth")
app.include_router(endpoints.router)
app.include_router(quantum_endpoints.router, prefix="/quantum")
app.include_router(websocket.router)
app.include_router(copilot_integration.router, prefix="/copilot")
app.include_router(jsonrpc.router, prefix="/jsonrpc")
app.include_router(void.router)
app.include_router(troubleshoot.router)
app.include_router(help.router)
app.include_router(comms_hub.router)
app.include_router(upload.router)
app.include_router(stream.router)

@app.get("/health")
async def health():
    return {"status": "ok"}
