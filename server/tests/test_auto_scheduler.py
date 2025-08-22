import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.automation.auto_scheduler import AutoScheduler


@pytest.fixture
def client():
    return TestClient(app)


def test_schedule_task():
    scheduler = AutoScheduler()
    task = scheduler.schedule_task("test_task", 60)
    assert task is not None
