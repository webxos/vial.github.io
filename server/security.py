import os
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from server.config import settings


ALLOWED_ORIGINS = [
    "https://vial.github.io",
    "https://your-app.netlify.app",
    "https://your-app.vercel.app",
    "http://localhost:8000"
]


def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def verify_jwt(token: str):
    try:
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET", settings.JWT_SECRET),
            algorithms=["RS256"],
            options={"verify_aud": False}
        )
        return payload
    except JWTError as e:
        raise JWTError(f"Token verification failed: {str(e)}")
