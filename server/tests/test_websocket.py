from fastapi.testclient import TestClient
from ..mcp_server import app


def test_websocket_connection():
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"vial_id": "vial1", "message": "test"})
        response = websocket.receive_json()
        assert response["status"] == "sent"
        assert response["vial_id"] == "vial1"
