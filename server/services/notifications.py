import aiohttp
from server.config import get_settings
from server.logging import logger

class NotificationService:
    def __init__(self):
        self.settings = get_settings()

    async def send_notification(self, user_id: str, message: str, method: str = "inapp"):
        try:
            if method == "inapp":
                logger.info(f"In-app notification to {user_id}: {message}")
                # Placeholder: Store notification in MongoDB or Redis
                return {"status": "notification sent", "method": "inapp", "user_id": user_id, "message": message}
            elif method == "email":
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.settings.NOTIFICATION_API_URL,
                        json={"user_id": user_id, "message": message}
                    ) as response:
                        logger.info(f"Email notification sent to {user_id}: {response.status}")
                        return {"status": "notification sent", "method": "email", "user_id": user_id}
            else:
                raise ValueError("Invalid notification method")
        except Exception as e:
            logger.error(f"Notification error: {str(e)}")
            raise ValueError(f"Notification failed: {str(e)}")

notification_service = NotificationService()
