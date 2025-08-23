from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from server.logging import logger
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.post("/auth/token")
async def generate_token(network_id: str, session_id: str):
    request_id = str(uuid.uuid4())
    try:
        secret_key = os.getenv("JWT_SECRET_KEY", "secret-token")
        token = jwt.encode(
            {"network_id": network_id, "session_id": session_id, "uuid": str(uuid.uuid4())},
            secret_key,
            algorithm="HS256"
        )
        logger.info(f"Generated token for {network_id}", request_id=request_id)
        return {"token": token, "request_id": request_id}
    except Exception as e:
        logger.error(f"Token generation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=401, detail=str(e))

@router.get("/auth/validate")
async def validate_token(token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        secret_key = os.getenv("JWT_SECRET_KEY", "secret-token")
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        logger.info(f"Validated token for {payload['network_id']}", request_id=request_id)
        return {"status": "valid", "request_id": request_id}
    except JWTError as e:
        logger.error(f"Token validation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=401, detail=str(e))
