import pytest
from fastapi.testclient import TestClient
from server.mcp_server import app
from server.services.notification import send_notification


client = TestClient(app)


@pytest.mark.asyncio
async def test_in_app_notification():
    response = await send_notification("Test in-app message", channel="in-app")
    assert response["status"] == "sent"
    assert response["message"] == "Test in-app message"


@pytest.mark.asyncio
async def test_invalid_channel_notification():
    with pytest.raises(Exception) as exc_info:
        await send_notification("Test message", channel="invalid")
    assert exc_info.value.status_code == 400
    assert "Invalid channel" in str(exc_info.value)
