```python
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def oauth_middleware(token: str = Depends(oauth2_scheme)):
    """Validate OAuth2.0 token"""
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"user_id": token[:8]}  # Mock token validation
```
