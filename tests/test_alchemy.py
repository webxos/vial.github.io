from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_quantum_link():
    response = client.post("/quantum/link", json={"node_a": "node1", "node_b": "node2"})
    assert response.status_code == 200
    assert "link_id" in response.json()
    assert "output" in response.json()

def test_generate_credentials():
    response = client.post("/generate-credentials")
    assert response.status_code == 200
    assert "token" in response.json()
