from fastapi import HTTPException
from jose import jwt, JWTError
from server.config import settings
from server.logging import logger
import uuid


async def map_oauth_to_mcp_session(token: str, request_id: str) -> dict:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user_id")
        mcp_session = {
            "session_id": str(uuid.uuid4()),
            "user_id": user_id,
            "scopes": payload.get("scopes", []),
            "created_at": datetime.utcnow().isoformat()
        }
        logger.log(f"Mapped OAuth token to MCP session for user: {user_id}", request_id=request_id)
        return mcp_session
    except JWTError as e:
        logger.log(f"JWT error during MCP session mapping: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=401, detail="Invalid token")
