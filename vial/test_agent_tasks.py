import pytest
from server.services.agent_tasks import AgentTasks
from server.services.error_logging import ErrorLogger
from server.logging_config import logger
import uuid

@pytest.fixture
def agent_tasks():
    return AgentTasks()

@pytest.fixture
def error_logger():
    return ErrorLogger()

@pytest.mark.asyncio
async def test_execute_task(agent_tasks):
    request_id = str(uuid.uuid4())
    result = await agent_tasks.execute_task(
        task_name="train_model",
        params={"vial_id": "vial1", "network_id": "54965687-3871-4f3d-a803-ac9840af87c4"},
        request_id=request_id
    )
    assert result["status"] == "trained"
    assert result["request_id"] == request_id
    prompt = await agent_tasks.get_pretrained_prompt("train_model")
    assert prompt["task_name"] == "train_model"
    logger.info("Task execution test passed", request_id=request_id)

@pytest.mark.asyncio
async def test_global_command(agent_tasks):
    request_id = str(uuid.uuid4())
    result = await agent_tasks.execute_global_command(
        command_name="sync_agents",
        params={"network_id": "54965687-3871-4f3d-a803-ac9840af87c4"},
        request_id=request_id
    )
    assert result["status"] == "coordinated"
    assert len(result["results"]) == 4
    logger.info("Global command test passed", request_id=request_id)

def test_error_logging(error_logger):
    request_id = str(uuid.uuid4())
    error_logger.log_error("Test error", request_id)
    logs = error_logger.get_logs(request_id)
    assert len(logs) == 1
    assert logs[0]["request_id"] == request_id
    assert logs[0]["message"] == "Test error"
    logger.info("Error logging test passed", request_id=request_id)
