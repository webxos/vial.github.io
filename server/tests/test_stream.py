import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.api.stream import StreamManager


@pytest.fixture
def client():
    return TestClient(app)


def test_stream_connection():
    client = TestClient(app)
    with client.websocket_connect("/stream") as websocket:
        websocket.send_json({"vial_id": "vial1"})
        data = websocket.receive_json()
        assert "vial_id" in data
        assert data["vial_id"] == "vial1"


def test_stream_data():
    client = TestClient(app)
    with client.websocket_connect("/stream") as websocket:
        websocket.send_json({"vial_id": "vial1"})
        data = websocket.receive_json()
        assert "state" in data
