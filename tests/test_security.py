from fastapi.testclient import TestClient
from server import app
from server.security import create_access_token

client = TestClient(app)

def test_token_creation():
    token = create_access_token({"sub": "testuser"})
    assert token is not None
    assert isinstance(token, str)

def test_auth_endpoint():
    token = create_access_token({"sub": "admin"})
    response = client.post("/auth/token", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
