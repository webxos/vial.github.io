from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from server.config import settings
from server.logging import logger

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def require_auth(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        logger.log(f"Authenticated user: {user_id}")
        return {"id": user_id}
    except JWTError as e:
        logger.log(f"JWT error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")


def require_visual_config_permission(action: str):
    def decorator(func):
        async def wrapper(token: str = Depends(oauth2_scheme), *args, **kwargs):
            user = require_auth(token)
            if action not in ["read", "write", "deploy"]:
                raise HTTPException(status_code=403, detail="Invalid action")
            # Placeholder for role-based access control
            if user["id"] != "admin":  # Simplified check
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            logger.log(f"Permission granted for {action} to user: {user['id']}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator
