from jose import jwt
from datetime import datetime, timedelta
from server.config.settings import settings
from server.services.advanced_logging import AdvancedLogger


logger = AdvancedLogger()


def refresh_token(token: str):
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        new_payload = payload.copy()
        new_payload["exp"] = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
        new_token = jwt.encode(new_payload, settings.jwt_secret, algorithm="HS256")
        logger.log("Token refreshed", extra={"username": payload.get("sub")})
        return {"access_token": new_token, "token_type": "bearer"}
    except Exception as e:
        logger.log("Token refresh failed", extra={"error": str(e)})
        return {"error": str(e)}
