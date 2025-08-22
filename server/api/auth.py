from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from datetime import datetime, timedelta
from server.config import settings
from server.security import require_auth
from server.logging import logger

router = APIRouter()


@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        # Simplified user validation (replace with real user DB check)
        if form_data.username != "test" or form_data.password != "test":
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token_data = {"sub": form_data.username, "exp": datetime.utcnow() + timedelta(hours=1)}
        token = jwt.encode(token_data, settings.JWT_SECRET, algorithm="HS256")
        logger.log(f"Generated token for user: {form_data.username}")
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.log(f"Token generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def get_current_user(user=Depends(require_auth)):
    try:
        logger.log(f"User profile accessed: {user['id']}")
        return {"user_id": user["id"]}
    except Exception as e:
        logger.log(f"User profile error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
