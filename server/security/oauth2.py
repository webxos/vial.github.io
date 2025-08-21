from fastapi import HTTPException
from server.models.auth_agent import auth_agent


async def verify_token(username: str, password: str):
    result = auth_agent.authenticate(username, password)
    if result["status"] == "authenticated":
        return "valid_token"
    raise HTTPException(status_code=401, detail="Invalid credentials")


async def get_current_user(token: str):
    if verify_token(token):
        return {"username": "authenticated_user"}
    raise HTTPException(status_code=401, detail="Invalid token")
