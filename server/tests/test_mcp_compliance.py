from fastapi.testclient import TestClient
from server.mcp_server import app
import pytest

client = TestClient(app)

def test_jsonrpc_compliance():
    response = client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": "status", "id": 1})
    assert response.status_code == 401  # Unauthorized without token
    
    # Placeholder: Add token-based test after login
    response = client.post("/jsonrpc", json={"jsonrpc": "2.0", "method": "help", "id": 2})
    assert response.status_code == 401

def test_oauth_compliance():
    response = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    
    response = client.post("/auth/token", data={"username": "invalid", "password": "wrong"})
    assert response.status_code == 401
