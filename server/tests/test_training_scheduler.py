import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_schedule_training():
    response = client.post("/agent/schedule", json={"task_id": "train_vial", "interval": 60,
                                                   "params": {"vial_id": "vial1"}})
    assert response.status_code == 200
    assert response.json()["status"] == "scheduled"
