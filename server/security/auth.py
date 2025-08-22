from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from server.config.settings import settings
from server.logging import logger

class AuthManager:
    def __init__(self):
        self.bearer = HTTPBearer()

    async def authenticate(self, credentials: HTTPAuthorizationCredentials = None):
        if not credentials:
            raise HTTPException(status_code=401, detail="No authentication provided")
        token = credentials.credentials
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
            return payload.get("user_id")
        except jwt.InvalidTokenError:
            logger.log("Invalid JWT token")
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

auth = AuthManager()
