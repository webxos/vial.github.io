from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from requests_oauthlib import OAuth2Session
from server.models.auth_agent import AuthAgent
import os


oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

class AuthManager:
    def __init__(self):
        self.client_id = os.getenv("GITHUB_CLIENT_ID")
        self.client_secret = os.getenv("GITHUB_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GITHUB_REDIRECT_URI")
        self.auth_agent = AuthAgent()

    async def authenticate(self, token: str = Depends(oauth2_scheme)):
        github = OAuth2Session(self.client_id, token={"access_token": token})
        user_info = github.get("https://api.github.com/user").json()
        if not user_info.get("id"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid GitHub authentication"
            )
        return self.auth_agent.process_user(user_info)


auth = AuthManager()
