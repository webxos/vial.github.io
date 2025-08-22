import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.error_recovery import ErrorRecovery
from unittest.mock import AsyncMock, patch


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_recover_task():
    recovery = ErrorRecovery(app)
    recovery.task_manager.execute_task = AsyncMock(side_effect=Exception("Test error"))
    response = await recovery.recover_task("train_vial", {"vial_id": "vial1"})
    assert response["status"] == "recovered"
    assert "Test error" in response["error"]
