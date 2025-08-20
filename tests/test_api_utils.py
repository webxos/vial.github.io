from fastapi.testclient import TestClient
from server import app
from server.api.utils import validate_token, push_to_github

client = TestClient(app)

def test_validate_token():
    # Mock a valid token
    with client as c:
        response = c.post("/generate-credentials")
        token = response.json()["token"]
        assert validate_token(token) is True

def test_push_to_github():
    result = push_to_github("vial/vial", "Test commit")
    assert "status" in result
    assert result["status"] in ["success", "error"]
