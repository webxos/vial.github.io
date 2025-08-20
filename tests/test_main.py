from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_mcp_tool():
    response = client.post("/mcp/tools/process", json={"data": "1 2 3 4 5 6 7 8 9 10"})
    assert response.status_code == 200
    assert "result" in response.json()
