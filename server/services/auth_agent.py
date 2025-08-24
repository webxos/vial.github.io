from jose import jwt, JWTError
from fastapi.security import OAuth2AuthorizationCodeBearer
from server.models.user_repository import UserRepository
from server.config.settings import load_settings
from sqlalchemy.orm import Session
import logging
import time

logging.basicConfig(level=logging.INFO, filename="logs/auth_agent.log")
settings = load_settings()

class AuthAgent:
    def __init__(self, db: Session):
        self.db = db
        self.user_repo = UserRepository(db)
        self.oauth2_scheme = OAuth2AuthorizationCodeBearer(
            authorizationUrl="/mcp/auth/authorize",
            tokenUrl="/mcp/auth/token",
            scheme_name="OAuth2PKCE"
        )

    async def authenticate_user(self, token: str) -> dict:
        """Authenticate user with JWT token."""
        try:
            # Validate JWT with Kyber-512 compatible secret
            payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
            wallet_id = payload.get("sub")
            if not wallet_id:
                raise ValueError("Invalid token: missing wallet_id")
            
            # Update user access time
            self.user_repo.update_access_time(wallet_id)
            return {"wallet_id": wallet_id, "role": payload.get("role", "user")}
        except JWTError as e:
            logging.error(f"JWT validation error: {str(e)}")
            raise ValueError(f"Invalid token: {str(e)}")
        except Exception as e:
            logging.error(f"Authentication error: {str(e)}")
            raise ValueError(f"Authentication failed: {str(e)}")

    async def create_token(self, wallet_id: str, role: str = "user") -> str:
        """Create JWT token for user."""
        try:
            payload = {
                "sub": wallet_id,
                "role": role,
                "exp": time.time() + 3600  # 1 hour expiry
            }
            token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
            self.user_repo.update_access_time(wallet_id)
            return token
        except Exception as e:
            logging.error(f"Token creation error: {str(e)}")
            raise ValueError(f"Token creation failed: {str(e)}")
