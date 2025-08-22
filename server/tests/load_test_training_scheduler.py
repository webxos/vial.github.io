import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.training_scheduler import TrainingScheduler
import asyncio
import time


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_load_training_scheduler():
    scheduler = TrainingScheduler(app)
    start_time = time.time()
    tasks = [scheduler.schedule_training("train_vial", 1, {"vial_id": f"vial{i}") 
             for i in range(10)]
    await asyncio.gather(*tasks)
    end_time = time.time()
    duration = end_time - start_time
    assert duration < 15, f"Load test exceeded 15s: {duration}s"
