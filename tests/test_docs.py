from fastapi.testclient import TestClient
from server.mcp_server import app

client = TestClient(app)


def test_openapi_schema():
    response = client.get("/docs/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Vial MCP Controller"
    assert schema["info"]["version"] == "2.7"
    assert "/jsonrpc" in schema["paths"]
    assert "/ws" in schema["paths"]
    assert "/docs/openapi.json" in schema["paths"]
