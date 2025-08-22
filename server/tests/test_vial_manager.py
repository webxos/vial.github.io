import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.models.vial_manager import VialManager


@pytest.fixture
def client():
    return TestClient(app)


def test_train_vial():
    vial_manager = VialManager()
    result = vial_manager.train_vial("vial1")
    assert result is not None


def test_authenticate_vial():
    vial_manager = VialManager()
    # Mock token_response as it’s undefined; assume a valid response for now
    result = vial_manager.authenticate_vial("vial1", "valid_token")
    assert result is True
