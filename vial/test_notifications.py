import pytest
from server.services.notifications import NotificationService
from server.logging_config import logger
import uuid

@pytest.fixture
def notification_service():
    return NotificationService()


@pytest.mark.asyncio
async def test_send_task_notification(notification_service):
    request_id = str(uuid.uuid4())
    task_name = "train_model"
    status = "success"
    await notification_service.send_task_notification(task_name, status, request_id)
    logger.info(f"Task notification test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_send_error_notification(notification_service):
    request_id = str(uuid.uuid4())
    task_name = "create_agent"
    error = "Task failed"
    await notification_service.send_task_notification(task_name, error, request_id)
    logger.info(f"Error notification test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_broadcast_notification(notification_service):
    request_id = str(uuid.uuid4())
    message = {"event": "system_update", "details": "Server restarted"}
    await notification_service.broadcast_notification(message, request_id)
    logger.info(f"Broadcast notification test passed", request_id=request_id)
