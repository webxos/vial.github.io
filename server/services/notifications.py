import httpx
from server.config import settings
from fastapi import HTTPException


async def send_notification(message: str, channel: str = "in-app"):
    async with httpx.AsyncClient() as client:
        try:
            if channel == "in-app":
                return {"status": "sent", "message": message}
            elif channel == "email":
                response = await client.post(
                    settings.NOTIFICATION_API_URL,
                    json={"message": message, "type": "email"}
                )
                response.raise_for_status()
                return {"status": "sent", "channel": channel}
            else:
                raise HTTPException(status_code=400, detail="Invalid channel")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
