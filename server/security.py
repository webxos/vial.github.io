from python_jose import jwt
from passlib.context import CryptContext
from server.logging import logger
import secrets


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def verify_token(token: str) -> bool:
    try:
        jwt.decode(token, "secret_key", algorithms=["HS256"])
        return True
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        return False


def generate_credentials() -> tuple:
    api_key = secrets.token_hex(16)
    api_secret = secrets.token_hex(32)
    return api_key, api_secret
