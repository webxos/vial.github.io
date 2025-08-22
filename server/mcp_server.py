from fastapi import FastAPI
from fastapi.security import OAuth2PasswordBearer
from server.services.advanced_logging import AdvancedLogger
from server.api import visual_router, webxos_wallet
from jose import jwt, JWTError


app = FastAPI()
logger = AdvancedLogger()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def validate_token(token: str):
    try:
        payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
        logger.log("Token validated",
                   extra={"user_id": payload.get("user_id")})
        return payload
    except JWTError:
        logger.log("Token validation failed",
                   extra={"error": "Invalid token"})
        raise


app.include_router(visual_router.router, prefix="/visual")
app.include_router(webxos_wallet.router, prefix="/wallet")
logger.log("MCP server initialized",
           extra={"version": "2.9.3"})
