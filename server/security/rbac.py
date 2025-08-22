from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from server.config.settings import settings
from server.services.advanced_logging import AdvancedLogger


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
logger = AdvancedLogger()


async def check_role(token: str = Depends(oauth2_scheme), required_role: str = "admin"):
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        role = payload.get("role", "user")
        if role != required_role:
            logger.log("RBAC check failed", extra={"role": role, "required_role": required_role})
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        logger.log("RBAC check passed", extra={"role": role})
        return payload
    except Exception as e:
        logger.log("RBAC error", extra={"error": str(e)})
        raise HTTPException(status_code=401, detail="Invalid token")
