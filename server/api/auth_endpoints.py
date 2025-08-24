```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
import logging
import jwt
from cryptography.hazmat.primitives.asymmetric import kyber
from server.config.settings import settings
from server.utils.security_sanitizer import sanitize_input

logger = logging.getLogger(__name__)
router = APIRouter()

class AuthRequest(BaseModel):
    wallet_id: str
    public_key: str

@router.post("/mcp/auth")
async def authenticate_wallet(request: AuthRequest) -> Dict:
    """Authenticate user with .mdwallet and issue JWT."""
    try:
        sanitized_wallet_id = sanitize_input(request.wallet_id)
        sanitized_public_key = sanitize_input(request.public_key)

        # Validate public key with Kyber-512
        try:
            kyber_key = kyber.Kyber512()
            public_key = bytes.fromhex(sanitized_public_key)
            # Mock validation (real implementation would verify key)
            if len(public_key) != kyber_key.public_key_length():
                raise ValueError("Invalid public key length")
        except Exception as e:
            logger.error(f"Kyber-512 validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid public key")

        # Generate JWT
        token = jwt.encode(
            {"wallet_id": sanitized_wallet_id, "role": "user"},
            settings.JWT_SECRET,
            algorithm="HS256"
        )
        logger.info(f"Authenticated wallet: {sanitized_wallet_id}")
        return {"access_token": token}
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```
