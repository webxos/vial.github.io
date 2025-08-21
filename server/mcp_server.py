from fastapi import FastAPI
from server.api import auth, endpoints, quantum_endpoints
from server.api.cache_control import cache_response
from server.api.rate_limiter import rate_limit
from server.security import setup_cors


app = FastAPI(
    title="Vial MCP Controller",
    description="Modular control plane for AI-driven task management",
    version="2.7.0",
)


app.middleware("http")(cache_response)
app.middleware("http")(rate_limit)
setup_cors(app)


app.include_router(auth.router, prefix="/auth")
app.include_router(endpoints.router)
app.include_router(quantum_endpoints.router, prefix="/quantum")


@app.get("/health")
async def health():
    return {"status": "ok"}
