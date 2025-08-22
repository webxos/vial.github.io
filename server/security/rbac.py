from fastapi import Depends, HTTPException
from server.security.auth import auth

def require_role(role: str):
    async def role_checker(user_id: str = Depends(auth.authenticate)):
        # Simulate role check from DB or config
        if role != "admin":
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user_id
    return role_checker
