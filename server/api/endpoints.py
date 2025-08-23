from fastapi import APIRouter, Depends, HTTPException
from server.services.agent_tasks import AgentTasks
from server.services.backup_restore import BackupRestore
from server.services.github_integration import GitHubIntegration
from server.services.notifications import NotificationService
from server.api.rate_limiter import RateLimiter
import uuid

router = APIRouter(prefix="/v1", tags=["v1"], dependencies=[Depends(RateLimiter().check_rate_limit)])

@router.post("/backup/wallet")
async def backup_wallet(network_id: str, backup_service: BackupRestore = Depends()):
    request_id = str(uuid.uuid4())
    return await backup_service.backup_wallet(network_id, request_id)

@router.post("/restore/wallet")
async def restore_wallet(backup_id: str, backup_service: BackupRestore = Depends()):
    request_id = str(uuid.uuid4())
    return await restore_wallet(backup_id, request_id)

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
