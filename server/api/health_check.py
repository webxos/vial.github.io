from fastapi import APIRouter, HTTPException
from server.mcp_server import app


router = APIRouter()

@router.get("/help")
async def get_help():
    help_text = """
    Available endpoints:
    - /health: Check server status
    - /auth/login: Start OAuth flow
    - /agent/task: Execute tasks
    """
    return {"help": help_text.strip()}


@router.get("/docs")
async def get_docs():
    return {"message": "See API.md for full documentation"}
