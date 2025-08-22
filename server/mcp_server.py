from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.api import (
    auth, endpoints, quantum_endpoints, websocket, 
    copilot_integration, jsonrpc, void, troubleshoot, 
    help, comms_hub, upload, stream, visual_router
)
from server.api.webxos_wallet import router as wallet_router
from server.api.cache_control import cache_response
from server.api.rate_limiter import rate_limit
from server.api.middleware import logging_middleware
from server.error_handler import setup_error_handlers
from server.security import setup_cors, setup_security_headers
from server.services.agent_tasks import setup_agent_tasks
from server.services.prompt_training import setup_prompt_training
from server.services.training_scheduler import setup_training_scheduler
from server.services.git_trainer import setup_git_trainer
from server.services.backup_restore import setup_backup_restore
from server.quantum.quantum_sync import setup_quantum_sync
from server.automation.deployment import setup_deployment
from server.automation.task_scheduler import setup_task_scheduler
from server.services.advanced_logging import AdvancedLogger
from server.config.settings import settings


app = FastAPI(
    title="Vial MCP Controller",
    description="Modular control plane for AI-driven task management",
    version="2.9.3"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.middleware("http")(cache_response)
app.middleware("http")(rate_limit)
app.middleware("http")(logging_middleware)


setup_error_handlers(app)
setup_cors(app)
setup_security_headers(app)
setup_agent_tasks(app)
setup_prompt_training(app)
setup_training_scheduler(app)
setup_git_trainer(app)
setup_backup_restore(app)
setup_quantum_sync(app)
setup_deployment(app)
setup_task_scheduler(app)


app.include_router(auth.router, prefix="/auth")
app.include_router(endpoints.router)
app.include_router(quantum_endpoints.router, prefix="/quantum")
app.include_router(websocket.router)
app.include_router(copilot_integration.router, prefix="/copilot")
app.include_router(jsonrpc.router)
app.include_router(void.router)
app.include_router(troubleshoot.router)
app.include_router(help.router)
app.include_router(comms_hub.router)
app.include_router(upload.router)
app.include_router(stream.router)
app.include_router(wallet_router, prefix="/wallet")
app.include_router(visual_router.router, prefix="/visual")


@app.on_event("startup")
async def startup_event():
    from server.services.vial_manager import VialManager
    app.state.vial_manager = VialManager()
    app.state.vial_manager.create_vial_agents()
    app.state.logger = AdvancedLogger()
    app.state.logger.log("Vial MCP Controller started with 4 agents", extra={"version": "2.9.3"})


@app.get("/")
async def root():
    return {"message": "Vial MCP Controller", "version": "2.9.3"}
