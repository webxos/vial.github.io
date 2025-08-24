import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_django_admin_success():
    """Test successful Django admin access."""
    with patch("server.api.django_integration.call_django_admin", new=AsyncMock()) as mock_call:
        mock_call.return_value = {"result": "admin response"}
        response = client.post(
            "/mcp/django/admin",
            json={"resource": "quantum", "action": "list", "data": {}}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "result": {"result": "admin response"}}

@pytest.mark.asyncio
async def test_django_admin_failure():
    """Test Django admin access failure."""
    with patch("server.api.django_integration.call_django_admin", new=AsyncMock()) as mock_call:
        mock_call.side_effect = Exception("Django error")
        response = client.post(
            "/mcp/django/admin",
            json={"resource": "quantum", "action": "list", "data": {}}
        )
        assert response.status_code == 500
        assert "Django error" in response.json()["detail"]
