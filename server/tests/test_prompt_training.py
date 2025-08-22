from fastapi.testclient import TestClient
from server.mcp_server import app

# ... (previous content assumed)

def test_prompt_training():
    client = TestClient(app)
    resp = client.post("/agent/train", json={"vial_id": "all"})
    assert resp.json()["status"] == "training_complete"
