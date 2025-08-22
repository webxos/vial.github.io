from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from server.security.auth import auth
from server.models.auth_agent import AuthAgent
from server.config.settings import settings

router = APIRouter()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token",
    scopes={"user": "user info"}
)

@router.get("/login")
async def login():
    return {"message": "Redirect to GitHub OAuth", "url": oauth2_scheme.authorizationUrl}

@router.get("/callback")
async def callback(code: str):
    try:
        user_data = await auth.authenticate(code)
        auth_agent = AuthAgent(**user_data)
        return {"user": auth_agent.dict()}
    except HTTPException as e:
        raise e
