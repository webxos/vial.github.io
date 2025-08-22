import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.webxos_wallet import WalletModel
from server.services.agent_tasks import AgentTaskManager
from unittest.mock import AsyncMock, patch


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_task_manager():
    manager = AgentTaskManager(app)
    manager.vial_manager.train_vial = AsyncMock()
    manager.wallet.update_balance = AsyncMock(return_value=WalletModel(user_id="test_user", balance=10.0))
    manager.git_trainer.commit_changes = AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_execute_train_vial(client, mock_task_manager):
    with patch("server.services.agent_tasks.AgentTaskManager", return_value=mock_task_manager):
        response = client.post("/agent/task", json={"task_id": "train_vial", "params": {"vial_id": "vial1",
                                                                                      "user_id": "test_user"}})
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        mock_task_manager.vial_manager.train_vial.assert_called_once()
        mock_task_manager.wallet.update_balance.assert_called_once()


@pytest.mark.asyncio
async def test_execute_git_commit(client, mock_task_manager):
    with patch("server.services.agent_tasks.AgentTaskManager", return_value=mock_task_manager):
        response = client.post("/agent/task", json={"task_id": "git_commit", "params": {"message": "test commit"}})
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        mock_task_manager.git_trainer.commit_changes.assert_called_once()
