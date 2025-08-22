import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.prompt_training import PromptTrainer
from unittest.mock import AsyncMock, patch


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_prompt_trainer():
    trainer = PromptTrainer(app)
    trainer.alchemist.process_prompt = AsyncMock(return_value={"output": "trained"})
    trainer.git_trainer.save_training_data = AsyncMock()
    return trainer


@pytest.mark.asyncio
async def test_train_prompt(client, mock_prompt_trainer):
    with patch("server.services.prompt_training.PromptTrainer", return_value=mock_prompt_trainer):
        response = client.post("/agent/train", json={"prompt_text": "Test {context}", 
                                                    "context": {"data": "test"}})
        assert response.status_code == 200
        assert response.json() == {"output": "trained"}
        mock_prompt_trainer.alchemist.process_prompt.assert_called_once()
        mock_prompt_trainer.git_trainer.save_training_data.assert_called_once()
