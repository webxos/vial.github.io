import pytest
from server.services.memory_manager import MemoryManager
from server.services.error_logging import ErrorLogger
from server.logging_config import logger
import uuid

@pytest.fixture
def memory_manager():
    return MemoryManager()

@pytest.fixture
def error_logger():
    return ErrorLogger()

@pytest.mark.asyncio
async def test_save_session(memory_manager):
    request_id = str(uuid.uuid4())
    session_data = {
        "menu_info": {"selectedTool": "agent", "guideStep": 1},
        "build_progress": ["vial1"],
        "quantum_logic": {"qubits": 4},
        "task_memory": ["train_model"]
    }
    result = await memory_manager.save_session("test_token", session_data, request_id)
    assert result["status"] == "saved"
    assert result["request_id"] == request_id
    session = await memory_manager.get_session("test_token", request_id)
    assert session["menu_info"] == session_data["menu_info"]
    logger.info("Session save test passed", request_id=request_id)

@pytest.mark.asyncio
async def test_reset_session(memory_manager):
    request_id = str(uuid.uuid4())
    await memory_manager.save_session("test_token", {"menu_info": {"selectedTool": "agent"}}, request_id)
    result = await memory_manager.reset_session("test_token", request_id)
    assert result["status"] == "reset"
    session = await memory_manager.get_session("test_token", request_id)
    assert session == {}
    logger.info("Session reset test passed", request_id=request_id)

@pytest.mark.asyncio
async def test_task_relationship(memory_manager):
    request_id = str(uuid.uuid4())
    relationship_data = {
        "quantum_logic": {"qubits": 4},
        "training_data": {"accuracy": 0.95},
        "related_tasks": ["train_model", "quantum_circuit"]
    }
    result = await memory_manager.save_task_relationship("train_model", relationship_data, request_id)
    assert result["status"] == "saved"
    logger.info("Task relationship test passed", request_id=request_id)

def test_error_logging(error_logger):
    request_id = str(uuid.uuid4())
    error_logger.log_error("Test memory error", request_id)
    logs = error_logger.get_logs(request_id)
    assert len(logs) == 1
    assert logs[0]["request_id"] == request_id
    assert logs[0]["message"] == "Test memory error"
    logger.info("Memory error logging test passed", request_id=request_id)
