from fastapi import HTTPException, Depends
from server.services.security import verify_jwt
from server.models.auth_agent import AuthAgent
from server.services.logging import Logger
from jose import jwt
from server.config import settings
from datetime import datetime, timedelta
import uuid


logger = Logger("auth_manager")


class AuthManager:
    async def authenticate(self, network_id: str, session_id: str):
        try:
            token = jwt.encode(
                {
                    "user_id": network_id,
                    "session_id": session_id,
                    "exp": datetime.utcnow() + timedelta(hours=1)
                },
                settings.JWT_SECRET,
                algorithm="HS256"
            )
            address = str(uuid.uuid4())
            await logger.info(f"Authenticated session: {session_id}")
            return token, address
        except Exception as e:
            await logger.error(f"Auth error: {str(e)}")
            os.makedirs("db", exist_ok=True)
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.utcnow().isoformat()}]** Auth error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

    async def validate_token(self, token: str):
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
            return payload["user_id"]
        except Exception as e:
            await logger.error(f"Token validation error: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid token")

    async def validate_session(self, token: str, network_id: str):
        user_id = await self.validate_token(token)
        if user_id != network_id:
            await logger.error(f"Session validation failed for network {network_id}")
            raise HTTPException(status_code=403, detail="Invalid network ID")
        return True

    async void_session(self, token: str):
        try:
            auth_agent = AuthAgent()
            payload = await verify_jwt({"scheme": "Bearer", "credentials": token})
            await auth_agent.assign_role(payload["user_id"], "user")
            await logger.info(f"Session voided for user {payload['user_id']}")
        except Exception as e:
            await logger.error(f"Void session error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
