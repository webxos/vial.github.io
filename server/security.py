from fastapi import Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from server.config import config
from server.logging import logger

class SecurityManager:
    def __init__(self):
        self.bearer = HTTPBearer()

    def authenticate(self, credentials: HTTPAuthorizationCredentials):
        token = credentials.credentials
        try:
            payload = jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])
            return payload.get("user_id")
        except jwt.InvalidTokenError:
            logger.log("Invalid JWT token")
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    def get_security_headers(self):
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block"
        }

def setup_cors(app):
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def setup_security_headers(app):
    security = SecurityManager()
    async def add_headers(request: Request, call_next):
        response = await call_next(request)
        headers = security.get_security_headers()
        for key, value in headers.items():
            response.headers[key] = value
        return response
    app.middleware("http")(add_headers)
