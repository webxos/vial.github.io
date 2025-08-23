import pytest
from server.services.agent_tasks import AgentTasks
from server.logging_config import logger
import uuid

@pytest.fixture
def agent_tasks():
    return AgentTasks()


@pytest.mark.asyncio
async def test_create_agent(agent_tasks):
    request_id = str(uuid.uuid4())
    params = {"vial_id": "vial1", "x_position": 10}
    result = await agent_tasks.execute_task("create_agent", params, request_id)
    assert result["status"] == "success"
    assert result["vial_id"] == "vial1"
    logger.info(f"Create agent test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_execute_svg_task(agent_tasks):
    request_id = str(uuid.uuid4())
    task = {"task_name": "create_endpoint", "params": {"vial_id": "vial2", "endpoint": "/v1/test"}}
    result = await agent_tasks.execute_svg_task(task, request_id)
    assert result["status"] == "success"
    assert result["endpoint"] == "/v1/test"
    logger.info(f"Execute SVG task test passed", request_id=request_id)


@pytest.mark.asyncio
async def test_train_model(agent_tasks):
    request_id = str(uuid.uuid4())
    params = {"vial_id": "vial3", "epochs": 10}
    result = await agent_tasks.execute_task("train_model", params, request_id)
    assert result["status"] == "trained"
    assert "accuracy" in result
    logger.info(f"Train model test passed", request_id=request_id)
