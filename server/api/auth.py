from fastapi import APIRouter, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from server.security import verify_jwt
from jose import jwt
from server.config import settings
import secrets


router = APIRouter()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


class Token(BaseModel):
    access_token: str
    token_type: str


class Credentials(BaseModel):
    key: str
    secret: str


@router.post("/token", response_model=Token)
async def login():
    token = jwt.encode(
        {"sub": "user"}, settings.JWT_SECRET, algorithm="RS256"
    )
    return {"access_token": token, "token_type": "bearer"}


@router.post("/generate-credentials", response_model=Credentials)
async def generate_credentials(token: str = Depends(oauth2_scheme)):
    try:
        await verify_jwt(token)
        key = secrets.token_hex(16)
        secret = secrets.token_hex(32)
        return {"key": key, "secret": secret}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
