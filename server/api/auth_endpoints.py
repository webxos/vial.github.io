from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import jwt, JWTError
import httpx
import os
import base64
import hashlib
from datetime import datetime, timedelta

router = APIRouter(prefix="/mcp/auth", tags=["auth"])
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
    scopes={"openid": "OpenID", "email": "Email", "profile": "Profile"}
)

async def verify_token(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, os.getenv("OAUTH_CLIENT_SECRET"), algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.get("/login")
async def login():
    code_verifier = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip("=")
    
    auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={os.getenv('OAUTH_CLIENT_ID')}&"
        f"redirect_uri={os.getenv('OAUTH_REDIRECT_URI')}&"
        f"response_type=code&"
        f"scope=openid%20email%20profile&"
        f"code_challenge={code_challenge}&"
        f"code_challenge_method=S256"
    )
    return {"auth_url": auth_url, "code_verifier": code_verifier}

@router.post("/token")
async def exchange_code(code: str, code_verifier: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": os.getenv("OAUTH_CLIENT_ID"),
                "client_secret": os.getenv("OAUTH_CLIENT_SECRET"),
                "code": code,
                "code_verifier": code_verifier,
                "grant_type": "authorization_code",
                "redirect_uri": os.getenv("OAUTH_REDIRECT_URI")
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange code")
        token_data = response.json()
        access_token = jwt.encode(
            {"sub": token_data["id_token"], "exp": datetime.utcnow() + timedelta(hours=1)},
            os.getenv("OAUTH_CLIENT_SECRET"),
            algorithm="HS256"
        )
        return {"access_token": access_token, "token_type": "bearer"}
