from server.services.notification import notification_service
from unittest.mock import AsyncMock, patch


async def test_inapp_notification():
    result = await notification_service.send_notification("test_user",
                                                         "Test message", "inapp")
    assert result["status"] == "notification sent"
    assert result["method"] == "inapp"
    assert result["user_id"] == "test_user"
    assert result["message"] == "Test message"


async def test_email_notification(mock_post):
    mock_post.return_value.__aenter__.return_value.status = 200
    result = await notification_service.send_notification("test_user",
                                                         "Test email", "email")
    assert result["status"] == "notification sent"
    assert result["method"] == "email"
    mock_post.assert_called_once()


async def test_invalid_notification_method():
    with pytest.raises(ValueError, match="Invalid notification method"):
        await notification_service.send_notification("test_user",
                                                    "Test message", "invalid")
