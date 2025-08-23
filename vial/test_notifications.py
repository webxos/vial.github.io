import pytest
from server.services.notifications import NotificationService
from server.services.error_logging import ErrorLogger
from server.logging_config import logger
import uuid

@pytest.fixture
def notification_service():
    return NotificationService()

@pytest.fixture
def error_logger():
    return ErrorLogger()

@pytest.mark.asyncio
async def test_task_notification(notification_service):
    request_id = str(uuid.uuid4())
    result = await notification_service.send_task_notification("train_model", "success", request_id)
    assert result["status"] == "notified"
    assert result["request_id"] == request_id
    logger.info("Task notification test passed", request_id=request_id)

def test_error_logging(error_logger):
    request_id = str(uuid.uuid4())
    error_logger.log_error("Test notification error", request_id)
    logs = error_logger.get_logs(request_id)
    assert len(logs) == 1
    assert logs[0]["request_id"] == request_id
    assert logs[0]["message"] == "Test notification error"
    logger.info("Notification error logging test passed", request_id=request_id)
