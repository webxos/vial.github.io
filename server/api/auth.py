from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from server.security import verify_token, generate_credentials

router = APIRouter()


@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    token = verify_token(form_data.username, form_data.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": token, "token_type": "bearer"}


@router.post("/generate-credentials")
async def generate_api_credentials(token: str = Depends(verify_token)):
    key, secret = generate_credentials()
    return {"key": key, "secret": secret}
