from fastapi.testclient import TestClient
from server.api.health_check import check_system_health
from server.mcp_server import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_system_health_check():
    result = check_system_health()
    assert result["status"] == "healthy"
    assert "services" in result
    assert result["services"]["mongodb"] in ["healthy", "unavailable"]
    assert result["services"]["redis"] in ["healthy", "unavailable"]
