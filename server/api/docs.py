from fastapi import APIRouter
from fastapi.openapi.utils import get_openapi

router = APIRouter()


def custom_openapi():
    from server.mcp_server import app
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes
    )


router.get("/openapi.json")(custom_openapi)
