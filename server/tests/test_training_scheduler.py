from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.training_scheduler import TrainingScheduler


@pytest.fixture
def client():
    return TestClient(app)


def test_schedule_training():
    scheduler = TrainingScheduler(app)
    response = client.post("/agent/schedule", json={"task_id": "train_vial", 
                                                   "interval": 60, 
                                                   "params": {"vial_id": "vial1"}})
    assert response.status_code == 200
    assert response.json()["status"] == "scheduled"
