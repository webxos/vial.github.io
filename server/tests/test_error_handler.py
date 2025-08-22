import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.api.error_handler import setup_error_handlers


@pytest.fixture
def client():
    return TestClient(app)


def test_error_handler():
    setup_error_handlers(app)
    response = client.get("/nonexistent")
    assert response.status_code == 404
