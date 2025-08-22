from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise ValueError("No token provided")
    return {"token": token, "user": "authenticated"}


@router.post("/login")
async def login():
    return {"access_token": "fake_token", "token_type": "bearer"}
