from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from server.logging import logger
from datetime import datetime, timedelta
import uuid


class SessionData(BaseModel):
    user_id: str
    scopes: list[str]
    expires_at: str


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token")


async def map_oauth_to_mcp_session(token: str, request_id: str) -> dict:
    try:
        # Simulate token validation (replace with real JWT decoding)
        user_id = str(uuid.uuid4())
        expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat()
        session = SessionData(
            user_id=user_id,
            scopes=["wallet:read", "wallet:export", "git:push", "vercel:deploy"],
            expires_at=expires_at
        )
        logger.log(f"Mapped OAuth token to MCP session for user: {user_id}",
                   request_id=request_id)
        return session.dict()
    except Exception as e:
        logger.log(f"Auth mapping error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=401, detail=str(e))
