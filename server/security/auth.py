from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from server.services.advanced_logging import AdvancedLogger
from jose import jwt, JWTError


logger = AdvancedLogger()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def setup_auth(app: FastAPI):
    @app.post("/token")
    async def login():
        token = jwt.encode({"user_id": "user123"}, "your-secret-key", algorithm="HS256")
        logger.log("Token generated",
                   extra={"user_id": "user123"})
        return {"access_token": token, "token_type": "bearer"}
