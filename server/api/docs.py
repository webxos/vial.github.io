from fastapi import APIRouter
from fastapi.openapi.utils import get_openapi
from server.mcp_server import app

router = APIRouter(prefix="/docs", tags=["docs"])

@router.get("/openapi.json")
async def get_openapi_schema():
    return get_openapi(
        title=app.title,
        version="2.7",
        description="Vial MCP Controller API, compliant with Anthropic MCP standards",
        routes=app.routes,
    )
