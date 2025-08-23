from fastapi import APIRouter, WebSocket, Depends
from server.services.agent_tasks import AgentTasks
from server.services.memory_manager import MemoryManager
from server.services.notifications import NotificationService
from server.api.rate_limiter import RateLimiter
from server.logging_config import logger
import json
import uuid

router = APIRouter()

@router.websocket("/mcp/ws")
async def websocket_endpoint(websocket: WebSocket, agent_tasks: AgentTasks = Depends(), memory_manager: MemoryManager = Depends(), notification_service: NotificationService = Depends(), rate_limiter: RateLimiter = Depends()):
    request_id = str(uuid.uuid4())
    await websocket.accept()
    try:
        await rate_limiter.check_rate_limit(websocket.query_params.get("token", "test_token"))
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            task_name = message.get("task_name")
            params = message.get("params", {})
            session_token = message.get("session_token", "test_token")

            # Save session data
            session_data = {
                "menu_info": params.get("menu_info", {}),
                "build_progress": params.get("build_progress", []),
                "quantum_logic": params.get("quantum_logic", {}),
                "task_memory": [task_name]
            }
            await memory_manager.save_session(session_token, session_data, request_id)

            # Execute task
            try:
                result = await agent_tasks.execute_task(task_name, params, request_id)
                await notification_service.send_task_notification(task_name, result["status"], request_id)
                await websocket.send_json({"result": result, "request_id": request_id, "session_token": session_token})
            except Exception as e:
                error_message = f"Task execution error: {str(e)}"
                await websocket.send_json({"error": error_message, "request_id": request_id})
                logger.error(error_message, request_id=request_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", request_id=request_id)
        await websocket.close()
