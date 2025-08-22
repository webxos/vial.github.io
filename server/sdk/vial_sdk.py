from server.mcp_server import app
from fastapi.testclient import TestClient


def get_sdk_client():
    return TestClient(app)
