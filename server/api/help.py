from fastapi import APIRouter
from server.services.advanced_logging import AdvancedLogger


router = APIRouter()
logger = AdvancedLogger()


@router.get("/commands")
async def get_commands():
    commands = {
        "help": "Display available commands",
        "clear": "Clear terminal output",
        "components": "List available components",
        "connections": "List active connections",
        "export": "Export configuration as SVG",
        "deploy": "Deploy configuration to GitHub Pages"
    }
    logger.log("Help commands requested", extra={"commands_count": len(commands)})
    return {"commands": commands}
