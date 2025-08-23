import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.memory_manager import MemoryManager
from server.services.session_tracker import SessionTracker
from server.logging_config import logger
import uuid

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def memory_manager():
    return MemoryManager()

@pytest.fixture
def session_tracker():
    return SessionTracker()

@pytest.mark.asyncio
async def test_agent_task_automation(client, memory_manager):
    request_id = str(uuid.uuid4())
    response = client.post(
        "/v1/execute_svg_task",
        json={
            "task_name": "create_agent",
            "params": {"vial_id": "vial5", "x_position": 0, "y_position": 0}
        },
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    session = await memory_manager.get_session("test_token", request_id)
    assert "vial5" in session.get("build_progress", [])
    logger.info("Agent task automation test passed", request_id=request_id)

@pytest.mark.asyncio
async def test_quantum_task_execution(client, memory_manager):
    request_id = str(uuid.uuid4())
    response = client.post(
        "/v1/quantum_circuit",
        json={"qubits": 4, "network_id": "54965687-3871-4f3d-a803-ac9840af87c4", "quantum_logic": {"gates": ["H", "CNOT"]}},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "circuit_built"
    quantum_logic = await memory_manager.get_quantum_logic("quantum_circuit", request_id)
    assert quantum_logic["gates"] == ["H", "CNOT"]
    logger.info("Quantum task execution test passed", request_id=request_id)

@pytest.mark.asyncio
async def test_session_activity_tracking(session_tracker):
    request_id = str(uuid.uuid4())
    activity = {"action": "drag_drop", "details": {"vial_id": "vial1", "x_position": 0, "y_position": 0}}
    result = await session_tracker.track_activity("test_token", activity, request_id)
    assert result["status"] == "tracked"
    session = await memory_manager.get_session("test_token", request_id)
    assert len(session.get("activities", [])) > 0
    assert session["activities"][-1]["action"] == "drag_drop"
    logger.info("Session activity tracking test passed", request_id=request_id)
