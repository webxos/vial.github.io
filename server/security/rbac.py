from fastapi import Depends, HTTPException
from server.services.advanced_logging import AdvancedLogger
from jose import jwt


logger = AdvancedLogger()


async def check_role(token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
    try:
        payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
        role = payload.get("role", "user")
        if role != "admin":
            logger.log("RBAC check failed",
                       extra={"user_id": payload.get("user_id")})
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        logger.log("RBAC check passed",
                   extra={"user_id": payload.get("user_id")})
        return payload
    except Exception as e:
        logger.log("RBAC error",
                   extra={"error": str(e)})
        raise
