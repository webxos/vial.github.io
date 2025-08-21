from fastapi import APIRouter, Depends, HTTPException
from server.api.health_check import full_health_check
from server.services.logging import Logger
from server.services.security import verify_jwt


router = APIRouter()
logger = Logger("troubleshoot")


@router.get("/troubleshoot")
async def troubleshoot(token: str = Depends(verify_jwt)):
    try:
        health_status = await full_health_check()
        logs = []
        if os.path.exists("db/errorlog.md"):
            with open("db/errorlog.md", "r") as f:
                logs = f.readlines()[-10:]  # Last 10 log entries
        await logger.info(f"Troubleshoot requested by user {token.get('user_id')}")
        return {"health": health_status, "recent_logs": logs}
    except Exception as e:
        await logger.error(f"Troubleshoot error: {str(e)}")
        os.makedirs("db", exist_ok=True)
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.utcnow().isoformat()}]** Troubleshoot error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
