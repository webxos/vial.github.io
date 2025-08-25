from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt
import jwks_rsa
import os
from typing import Dict

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
    auto_error=False
)

class AuthService:
    def __init__(self):
        self.jwks_client = jwks_rsa.JwksClient(os.getenv("JWKS_URI", "https://www.googleapis.com/oauth2/v3/certs"))
        self.client_id = os.getenv("OAUTH_CLIENT_ID")
        self.client_secret = os.getenv("OAUTH_CLIENT_SECRET")
        self.redirect_uri = os.getenv("OAUTH_REDIRECT_URI")

    async def verify_token(self, token: str = Depends(oauth2_scheme)) -> Dict:
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")
        try:
            # Decode and verify JWT
            header = jwt.get_unverified_header(token)
            jwk = self.jwks_client.get_signing_key(header["kid"]).to_dict()
            payload = jwt.decode(
                token,
                jwk,
                algorithms=["RS256"],
                audience=self.client_id,
                issuer="https://accounts.google.com"
            )
            return payload
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

auth_service = AuthService()

router = APIRouter(prefix="/mcp/auth", tags=["auth"])

@router.get("/callback")
async def auth_callback(code: str):
    try:
        # Exchange code for token (simplified for example)
        return {"message": "Token exchanged successfully", "code": code}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Auth callback failed: {str(e)}")

@router.get("/verify")
async def verify(payload: Dict = Depends(auth_service.verify_token)):
    return {"message": "Token verified", "payload": payload}
