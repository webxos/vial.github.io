from fastapi import APIRouter, Depends, HTTPException
from server.models.auth_agent import AuthAgent
from server.automation.task_queue import TaskQueue
from server.services.logging import Logger


router = APIRouter()
logger = Logger("void")


@router.post("/void")
async def void_network(token: str = Depends(verify_jwt)):
    try:
        auth_agent = AuthAgent()
        task_queue = TaskQueue()
        await auth_agent.check_role(token.get("user_id"), "admin")
        await auth_agent.assign_role(token.get("user_id"), "user")  # Reset role
        await task_queue.add_task({"type": "reset", "params": {}})
        await logger.info(f"Network voided for user {token.get('user_id')}")
        return {"status": "voided"}
    except Exception as e:
        await logger.error(f"Void error: {str(e)}")
        os.makedirs("db", exist_ok=True)
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.utcnow().isoformat()}]** Void error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
