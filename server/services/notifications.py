from server.config import get_settings
from server.logging import logger
import aiohttp


class NotificationService:
    def __init__(self):
        self.settings = get_settings()

    async def send_notification(self, user_id: str, message: str, method: str):
        try:
            if method == "inapp":
                logger.info(f"Sent in-app notification to {user_id}: {message}")
                return {
                    "status": "notification sent",
                    "method": method,
                    "user_id": user_id,
                    "message": message
                }
            elif method == "email":
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.settings.NOTIFICATION_API_URL,
                        json={"user_id": user_id, "message": message}
                    ) as response:
                        response.raise_for_status()
                        logger.info(f"Sent email notification to {user_id}")
                        return {"status": "notification sent", "method": method}
            else:
                raise ValueError("Invalid notification method")
        except Exception as e:
            logger.error(f"Notification failed: {str(e)}")
            raise ValueError(f"Notification failed: {str(e)}")


notification_service = NotificationService()
