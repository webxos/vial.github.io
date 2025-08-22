from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from server.config.settings import settings
from server.services.advanced_logging import AdvancedLogger
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
logger = AdvancedLogger()


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm="HS256")
    return encoded_jwt


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != "admin" or form_data.password != "secret":
        logger.log("Authentication failed", extra={"username": form_data.username})
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": form_data.username})
    logger.log("Token generated", extra={"username": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/verify")
async def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        logger.log("Token verified", extra={"username": username})
        return {"username": username}
    except JWTError:
        logger.log("Token verification failed", extra={"token": token})
        raise HTTPException(status_code=401, detail="Invalid token")
