from fastapi import APIRouter
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html


router = APIRouter()


@router.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Vial MCP Controller",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@4/swagger-ui.css"
    )


@router.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Vial MCP Controller",
        redoc_js_url="https://unpkg.com/redoc@next/bundles/redoc.standalone.js"
    )
