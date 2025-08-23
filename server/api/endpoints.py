from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from server.services.agent_tasks import AgentTasks
from server.services.backup_restore import BackupRestore
from server.services.github_integration import GitHubIntegration
from server.services.memory_manager import MemoryManager
from server.services.notifications import NotificationService
from server.api.rate_limiter import RateLimiter
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1", tags=["v1"], dependencies=[Depends(RateLimiter().check_rate_limit)])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.post("/auth/token")
async def generate_token(network_id: str, session_id: str, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        token = f"token_{uuid.uuid4()}"
        await memory_manager.save_session(token, {"network_id": network_id, "session_id": session_id}, request_id)
        logger.info(f"Generated token for network_id {network_id}", request_id=request_id)
        return {"token": token, "request_id": request_id}
    except Exception as e:
        logger.error(f"Token generation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/auth/validate")
async def validate_token(token: str = Depends(oauth2_scheme), memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        session = await memory_manager.get_session(token, request_id)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid token")
        logger.info(f"Validated token {token}", request_id=request_id)
        return {"status": "valid", "request_id": request_id}
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backup/wallet")
async def backup_wallet(network_id: str, backup_service: BackupRestore = Depends()):
    request_id = str(uuid.uuid4())
    return await backup_service.backup_wallet(network_id, request_id)

@router.post("/restore/wallet")
async def restore_wallet(backup_id: str, backup_service: BackupRestore = Depends()):
    request_id = str(uuid.uuid4())
    return await backup_service.restore_wallet(backup_id, request_id)

@router.post("/backup/agent_config")
async def backup_agent_config(backup_service: BackupRestore = Depends()):
    request_id = str(uuid.uuid4())
    return await backup_service.backup_agent_config(request_id)

@router.post("/fork_repository")
async def fork_repository(github_service: GitHubIntegration = Depends()):
    request_id = str(uuid.uuid4())
    return await github_service.fork_repository(request_id)

@router.post("/commit_training")
async def commit_training(vial_id: str, github_service: GitHubIntegration = Depends()):
    request_id = str(uuid.uuid4())
    return await github_service.commit_training_results(vial_id, request_id)

@router.post("/execute_svg_task")
async def execute_svg_task(task: dict, agent_tasks: AgentTasks = Depends(), notification_service: NotificationService = Depends()):
    request_id = str(uuid.uuid4())
    try:
        result = await agent_tasks.execute_svg_task(task, request_id)
        await notification_service.send_task_notification(task["task_name"], result["status"], request_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quantum_circuit")
async def quantum_circuit(params: dict, agent_tasks: AgentTasks = Depends(), notification_service: NotificationService = Depends()):
    request_id = str(uuid.uuid4())
    try:
        result = await agent_tasks.execute_task("quantum_circuit", params, request_id)
        await notification_service.send_task_notification("quantum_circuit", result["status"], request_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save_session")
async def save_session(session_data: dict, token: str = Depends(oauth2_scheme), memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        result = await memory_manager.save_session(token, session_data, request_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/troubleshoot/status")
async def troubleshoot_status(token: str = Depends(oauth2_scheme), memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        result = await memory_manager.reset_session(token, request_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
