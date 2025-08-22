import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.training_scheduler import TrainingScheduler
from unittest.mock import AsyncMock, patch
import asyncio


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_schedule_training_load():
    scheduler = TrainingScheduler(app)
    scheduler.task_manager.execute_task = AsyncMock()
    scheduler.prompt_trainer.train_prompt = AsyncMock()
    await scheduler.schedule_training("train_vial", 1, {"vial_id": "vial1"})
    await asyncio.sleep(2)
    scheduler.task_manager.execute_task.assert_called()
    scheduler.prompt_trainer.train_prompt.assert_called()
