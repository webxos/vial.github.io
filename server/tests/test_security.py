import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.security import verify_jwt
from jose import jwt
from server.config import settings


client = TestClient(app)


@pytest.mark.asyncio
async def test_verify_jwt_valid():
    token = jwt.encode({"user_id": "test_user"}, settings.JWT_SECRET, algorithm="HS256")
    payload = await verify_jwt(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    assert payload["user_id"] == "test_user"


@pytest.mark.asyncio
async def test_verify_jwt_invalid():
    with pytest.raises(HTTPException) as exc:
        await verify_jwt(HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid_token"))
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid token"


@pytest.mark.asyncio
async def test_cors():
    response = await client.options(
        "/health",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"}
    )
    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" in response.headers
    assert response.headers["Access-Control-Allow-Origin"] == "*"
